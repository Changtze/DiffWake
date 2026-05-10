"""
    FlowField

Atmospheric / flow-field state container.  Mirrors the Flax `FlowField`
dataclass from `flow_field.py`.
"""
Base.@kwdef mutable struct FlowField
    wind_speeds::Vector{Float64}
    wind_directions::Vector{Float64}
    wind_shear::Float64
    wind_veer::Float64
    air_density::Float64
    turbulence_intensities::Vector{Float64}
    reference_wind_height::Float64

    u_initial_sorted::Union{Nothing, Array{Float64,4}}      = nothing
    v_initial_sorted::Union{Nothing, Array{Float64,4}}      = nothing
    w_initial_sorted::Union{Nothing, Array{Float64,4}}      = nothing
    dudz_initial_sorted::Union{Nothing, Array{Float64,4}}   = nothing

    u_sorted::Union{Nothing, Array{Float64,4}}  = nothing
    v_sorted::Union{Nothing, Array{Float64,4}}  = nothing
    w_sorted::Union{Nothing, Array{Float64,4}}  = nothing

    u::Union{Nothing, Array{Float64,4}}     = nothing
    v::Union{Nothing, Array{Float64,4}}     = nothing
    w::Union{Nothing, Array{Float64,4}}     = nothing
    dudz::Union{Nothing, Array{Float64,4}}  = nothing

    turbulence_intensity_field_sorted::Union{Nothing, Array{Float64,4}}     = nothing
    turbulence_intensity_field_sorted_avg::Union{Nothing, Array{Float64,4}} = nothing
    turbulence_intensity_field::Union{Nothing, Array{Float64,4}}            = nothing

    grid_resolution::Union{Nothing, Int} = nothing
    n_turbines::Union{Nothing, Int}      = nothing
    n_findex::Union{Nothing, Int}        = nothing
end

"""
    initialize_velocity_field!(ff::FlowField, grid::TurbineGrid)

Fills `ff` with the initial velocity field using a power-law shear profile.
"""
function initialize_velocity_field!(ff::FlowField, grid::TurbineGrid)
    z_sorted = grid.z_sorted
    B, T, Ny, Nz = size(z_sorted)
    safe_z = max.(z_sorted, 1e-7)

    wind_profile_plane = (safe_z ./ ff.reference_wind_height) .^ ff.wind_shear
    dudz_profile = (
        ff.wind_shear .*
        (1.0 / ff.reference_wind_height) ^ ff.wind_shear .*
        safe_z .^ (ff.wind_shear - 1.0)
    )

    # Expand wind speeds: (B,) → (B,1,1,1)
    ws = reshape(ff.wind_speeds, B, 1, 1, 1)

    u_init = ws .* wind_profile_plane
    dudz   = ws .* dudz_profile
    zeros4 = zeros(B, T, Ny, Nz)

    # Un-sort for the inertial-frame copies
    idxer = grid.unsorted_indices
    u_uns    = _take_along_axis2(u_init, idxer)
    v_uns    = _take_along_axis2(zeros4, idxer)
    w_uns    = _take_along_axis2(zeros4, idxer)
    dudz_uns = _take_along_axis2(dudz,   idxer)

    # TI field (B, T, 1, 1)
    turb_exp = zeros(B, T, 1, 1)
    for b in 1:B
        turb_exp[b, :, 1, 1] .= ff.turbulence_intensities[b]
    end

    ff.u_initial_sorted      = u_init
    ff.v_initial_sorted      = copy(zeros4)
    ff.w_initial_sorted      = copy(zeros4)
    ff.dudz_initial_sorted   = dudz

    ff.u_sorted = copy(u_init)
    ff.v_sorted = copy(zeros4)
    ff.w_sorted = copy(zeros4)

    ff.u    = u_uns
    ff.v    = v_uns
    ff.w    = w_uns
    ff.dudz = dudz_uns

    ff.turbulence_intensity_field_sorted = turb_exp
    ff.turbulence_intensity_field        = copy(turb_exp)

    ff.grid_resolution = Ny
    ff.n_turbines      = T
    ff.n_findex        = B

    return ff
end

"""
    finalize!(ff::FlowField, unsorted_indices)

Un-sort the solved fields back to the original turbine ordering.
"""
function finalize!(ff::FlowField, unsorted_indices::Array{Int,4})
    ff.u = _take_along_axis2(ff.u_sorted, unsorted_indices)
    ff.v = _take_along_axis2(ff.v_sorted, unsorted_indices)
    ff.w = _take_along_axis2(ff.w_sorted, unsorted_indices)

    ti_uns = _take_along_axis2(ff.turbulence_intensity_field_sorted, unsorted_indices)
    B, T, Ny, Nz = size(ti_uns)
    ti_avg = zeros(B, T, 1, 1)
    for b in 1:B, t in 1:T
        ti_avg[b, t, 1, 1] = mean(ti_uns[b, t, :, :])
    end
    ff.turbulence_intensity_field = ti_avg
    return ff
end
