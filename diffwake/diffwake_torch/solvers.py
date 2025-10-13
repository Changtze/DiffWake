from __future__ import annotations
import torch
import torch.nn as nn
from .turbine.turbine import thrust_coefficient, axial_induction
from .wake_deflection.gauss import calculate_transverse_velocity, wake_added_yaw, yaw_added_turbulence_mixing
from .utils import average_velocity
 

def sigmoid_step(x: torch.Tensor, edge: torch.Tensor, inv_w: float = 1.) -> torch.Tensor:
    # (x-edge)/width == (x-edge)*inv_w
    return ( (x - edge) * inv_w ).sigmoid()

def smooth_box(x, centre, half, inv_w=2.0):          # width≈1/0.5
    return sigmoid_step(x, centre-half, inv_w) * (1. - sigmoid_step(x, centre+half, inv_w))

def make_cc_buffers(grid, flow_field):
    B, T, Ny, Nz = grid.x_sorted.shape
    dtype  = flow_field.u_initial_sorted.dtype
    device = flow_field.u_initial_sorted.device

    v_wake_buf      = torch.zeros((B, T, Ny, Nz), dtype=dtype, device=device)
    w_wake_buf      = torch.zeros_like(v_wake_buf)
    turb_u_wake_buf = torch.zeros_like(v_wake_buf)
    Ctmp_buf        = torch.zeros((T, B, T, Ny, Nz), dtype=dtype, device=device)

    return dict(
        v_wake=v_wake_buf,
        w_wake=w_wake_buf,
        turb_u_wake=turb_u_wake_buf,
        Ctmp=Ctmp_buf,
    )






def cc_solver(farm, flow_field, grid, model_manager):
    buffers = make_cc_buffers(grid, flow_field)
    return cc_solver_(farm, flow_field, grid, model_manager, **buffers)

def cc_solver_(farm, flow_field, grid, model_manager,
               v_wake, w_wake, turb_u_wake, Ctmp ):
    B, T, Ny, Nz = grid.x_sorted.shape
    device = v_wake.device
    dtype  = v_wake.dtype

    v_wake.zero_()
    w_wake.zero_()
    turb_u_wake.zero_()
    Ctmp.zero_()
 
    turb_inflow_field = flow_field.u_initial_sorted.clone() 
    turbine_turbulence_intensity = flow_field.turbulence_intensities\
                                    .view(B, 1, 1, 1).repeat(1, T, 3, 3)  # contiguous
    ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]

    hub_height_i = torch.as_tensor(farm.hub_height, dtype=dtype, device=device).view(1,1,1,1)
    TSR_i = torch.as_tensor(farm.TSR,        dtype=dtype, device=device).view(1,1,1,1)

    x_c = grid.x_sorted.mean((2, 3), keepdim=True)
    y_c = grid.y_sorted.mean((2, 3), keepdim=True)
    z_c = grid.z_sorted.mean((2, 3), keepdim=True)
    rotor_diameter_i = torch.as_tensor(farm.rotor_diameter, dtype=dtype, device=device)
    for i in range(T):
        x_i = x_c[:, i:i+1]          # shape (B,1,1,1)
        y_i = y_c[:, i:i+1]
        z_i = z_c[:, i:i+1]
        
        mask = (
            (grid.x_sorted < x_i + 0.01) &
            (grid.x_sorted > x_i - 0.01) &
            (grid.y_sorted < y_i + 0.51 * rotor_diameter_i) &
            (grid.y_sorted > y_i - 0.51 * rotor_diameter_i)
        )

        
        turb_inflow_field = turb_inflow_field.clone()
        turb_inflow_field.masked_scatter_(mask, flow_field.u_initial_sorted[mask] - turb_u_wake[mask])
        
        turb_avg_vels = average_velocity(turb_inflow_field)[:,:,None,None]
        
        turb_Cts = thrust_coefficient(
            velocities=turb_avg_vels,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            thrust_fn=farm.thrust_coefficient_function,
            tilt_interp=farm.tilt_interp,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt,
            power_thrust_table=farm.power_thrust_table,
            average_method=grid.average_method,
        )
        
        turb_Cts = turb_Cts.view(B, T, 1, 1)
        turb_aIs = axial_induction(
            turb_avg_vels,
            flow_field.turbulence_intensity_field_sorted,
            flow_field.air_density,
            farm.yaw_angles_sorted,
            farm.tilt_angles_sorted,
            farm.power_setpoints_sorted,
            farm.axial_induction_function,
            tilt_interp=farm.tilt_interp,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt,   
            turbine_power_thrust_table=farm.power_thrust_table,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=None,
        )

        turb_aIs = turb_aIs
        turb_aIs = turb_aIs[:, :, None, None]
        u_i = turb_inflow_field[:, i:i+1]
        v_i = flow_field.v_sorted[:, i:i+1]
        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            axial_induction_function=farm.axial_induction_function,
            tilt_interp=farm.tilt_interp,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt,
            turbine_power_thrust_table=farm.power_thrust_table,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=None,
        )

        axial_induction_i = axial_induction_i[:, :, None, None]
        
        turbulence_intensity_i = turbine_turbulence_intensity[:, i:i+1]
        yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i.clone(),
                v_i.clone(),
                flow_field.u_initial_sorted,
                grid.y_sorted[:, i:i+1] - y_i,
                grid.z_sorted[:, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                turb_Cts[:, i:i+1],
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
                scale=2.0,
            )
            effective_yaw_i = yaw_angle_i + added_yaw
        else:
            effective_yaw_i = yaw_angle_i

        deflection_field = model_manager.deflection_model(
            x_i, 
            y_i, 
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, i:i+1],
            rotor_diameter_i,
            x = grid.x_sorted,
            y =  grid.y_sorted,
            z =  grid.z_sorted,
            U_free =  flow_field.u_initial_sorted,
            wind_veer = flow_field.wind_veer
        )
        if model_manager.enable_transverse_velocities:

            v_wake, w_wake = calculate_transverse_velocity(
                u_i, 
                flow_field.u_initial_sorted, 
                flow_field.dudz_initial_sorted,
                grid.x_sorted - x_i, 
                grid.y_sorted - y_i, 
                grid.z_sorted,
                rotor_diameter_i, 
                hub_height_i, 
                yaw_angle_i,
                turb_Cts[:, i:i+1], 
                TSR_i, 
                axial_induction_i,
                flow_field.wind_shear, 
                scale=2.0
            )



        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i, 
                turbulence_intensity_i, 
                v_i,
                flow_field.w_sorted[:, i:i+1],
                v_wake[:, i:i+1], 
                w_wake[:, i:i+1]
            )
            turbine_turbulence_intensity[:, i:i+1] = turbulence_intensity_i + 1. * I_mixing

        turb_u_wake,Ctmp =  model_manager.velocity_model(
            i, 
            x_i, y_i, z_i, u_i, 
            deflection_field,   # ok
            yaw_angle_i,
            turbine_turbulence_intensity, #ok
            turb_Cts,
            rotor_diameter_i,
            turb_u_wake, 
            Ctmp,
            x = grid.x_sorted, y =  grid.y_sorted, z =  grid.z_sorted,
            u_initial = flow_field.u_initial_sorted,
        )


        wake_added_turbulence_intensity  = model_manager.turbulence_model(
            ambient_turbulence_intensities,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            turb_aIs
        )

        area_overlap = 1.0 - (
            (turb_u_wake <= 0.05).sum(dim=(2, 3), keepdim=False).float()
            / (grid.grid_resolution * grid.grid_resolution)
        )

    
        area_overlap = area_overlap[:, :, None, None]


        downstream_influence_length = 15.0 * rotor_diameter_i
        eps = 1e-10
        downstream_start = grid.x_sorted > x_i + eps
        downstream_end = (grid.x_sorted <= downstream_influence_length + x_i - eps)#1.0 - smooth_step(grid.x_sorted, x_i + downstream_influence_length - eps, width=0.5)

        dy = torch.abs(grid.y_sorted - y_i)

        lateral_mask = dy < 2 * rotor_diameter_i - eps#1.0 - smooth_step(dy, 2 * rotor_diameter_i, width=0.2)
        wake_ti = torch.nan_to_num(wake_added_turbulence_intensity, nan=0.0, posinf=0.0)

        
        ti_added = (area_overlap * 
                    wake_ti * 
                    downstream_start * 
                    downstream_end * 
                    lateral_mask)

        turbine_turbulence_intensity = torch.maximum(
                torch.sqrt(ti_added**2 + ambient_turbulence_intensities**2), turbine_turbulence_intensity
            ) 

        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake


    flow_field.u_sorted = turb_inflow_field.clone()
    flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
    flow_field.turbulence_intensity_field_sorted_avg = torch.mean(
        turbine_turbulence_intensity,
        axis=(2,3)
    )[:, :, None, None]



import torch
from functorch.experimental.control_flow import cond   # for trace-friendly ifs


from .turbine.turbine import thrust_coefficient, axial_induction
from .wake_deflection.gauss import (
    calculate_transverse_velocity, wake_added_yaw, yaw_added_turbulence_mixing
)


def smooth_step(x: torch.Tensor, edge: float, width: float = 1.0) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-(x - edge) / width))

def smooth_box(val: torch.Tensor, ctr: float, half: float, width: float = 0.5) -> torch.Tensor:
    return smooth_step(val, ctr - half, width) * (1 - smooth_step(val, ctr + half, width))


