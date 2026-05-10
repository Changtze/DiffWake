"""
    TurbineGrid

Holds the rotated and wind-direction-sorted turbine grid coordinates.

Dimensions convention:
  - `B` = number of wind conditions (findex)
  - `T` = number of turbines
  - `Ny, Nz` = grid resolution in y and z on each rotor disc
"""
struct TurbineGrid
    turbine_coordinates::Matrix{Float64}   # (T, 3)
    turbine_diameter::Float64
    wind_directions::Vector{Float64}       # (B,)

    B::Int
    T::Int
    x_center_of_rotation::Float64
    y_center_of_rotation::Float64

    x::Matrix{Float64}          # (B, T) rotated
    y::Matrix{Float64}
    z::Matrix{Float64}

    x_sorted::Array{Float64,4}  # (B, T, Ny, Nz)
    y_sorted::Array{Float64,4}
    z_sorted::Array{Float64,4}

    sorted_indices::Array{Int,4}         # (B, T, Ny, Nz)
    sorted_coord_indices::Matrix{Int}    # (B, T)
    unsorted_indices::Array{Int,4}       # (B, T, Ny, Nz)

    radius::Vector{Float64}                  # (T,)
    x_sorted_inertial_frame::Array{Float64,4}
    y_sorted_inertial_frame::Array{Float64,4}
    z_sorted_inertial_frame::Array{Float64,4}

    grid_resolution::Int
    cubature_weights::Nothing   # placeholder
end

"""
    create_turbine_grid(coords, D, wind_dirs; grid_resolution=5)

Factory method – mirrors `TurbineGrid.create(...)` from the JAX code.
"""
function create_turbine_grid(
    turbine_coordinates::Matrix{Float64},   # (T, 3)
    turbine_diameter::Float64,
    wind_directions::Vector{Float64};       # (B,) in radians
    grid_resolution::Int = 5,
)
    B = length(wind_directions)
    T = size(turbine_coordinates, 1)

    xc = 0.5 * (minimum(turbine_coordinates[:, 1]) + maximum(turbine_coordinates[:, 1]))
    yc = 0.5 * (minimum(turbine_coordinates[:, 2]) + maximum(turbine_coordinates[:, 2]))

    x_rel = turbine_coordinates[:, 1] .- xc   # (T,)
    y_rel = turbine_coordinates[:, 2] .- yc
    z_rel = turbine_coordinates[:, 3]

    theta = wind_directions .- 270.0 * π / 180.0   # (B,)

    # Rotated coordinates  (B, T)
    x_rot = zeros(B, T)
    y_rot = zeros(B, T)
    z_rot = zeros(B, T)
    for b in 1:B
        ct, st = cos(theta[b]), sin(theta[b])
        for t in 1:T
            x_rot[b, t] = x_rel[t] * ct - y_rel[t] * st + xc
            y_rot[b, t] = x_rel[t] * st + y_rel[t] * ct + yc
            z_rot[b, t] = z_rel[t]
        end
    end

    radius = turbine_diameter * 0.5 * 0.5
    span   = range(-1.0, 1.0, length = grid_resolution)
    dy     = collect(span) .* radius
    dz     = collect(span) .* radius

    # Build 4-D grids  (B, T, Ny, Nz)
    Ny = Nz = grid_resolution
    x_grid = zeros(B, T, Ny, Nz)
    y_grid = zeros(B, T, Ny, Nz)
    z_grid = zeros(B, T, Ny, Nz)
    for b in 1:B, t in 1:T, iy in 1:Ny, iz in 1:Nz
        x_grid[b, t, iy, iz] = x_rot[b, t]
        y_grid[b, t, iy, iz] = y_rot[b, t] + dy[iy]
        z_grid[b, t, iy, iz] = z_rot[b, t] + dz[iz]
    end

    # Sort turbines by x along axis 2 (turbine axis)
    sorted_idx   = zeros(Int, B, T, Ny, Nz)
    unsorted_idx = zeros(Int, B, T, Ny, Nz)
    sorted_coord = zeros(Int, B, T)

    for b in 1:B
        perm = sortperm(x_rot[b, :])          # sort by x for this wind direction
        inv_perm = invperm(perm)
        sorted_coord[b, :] .= perm
        for iy in 1:Ny, iz in 1:Nz
            sorted_idx[b, :, iy, iz]   .= perm
            unsorted_idx[b, :, iy, iz] .= inv_perm
        end
    end

    x_sorted = _take_along_axis2(x_grid, sorted_idx)
    y_sorted = _take_along_axis2(y_grid, sorted_idx)
    z_sorted = _take_along_axis2(z_grid, sorted_idx)

    # Inertial frame back-rotation
    x_si = zeros(B, T, Ny, Nz)
    y_si = zeros(B, T, Ny, Nz)
    z_si = copy(z_sorted)
    for b in 1:B
        ct, st = cos(theta[b]), sin(theta[b])
        for t in 1:T, iy in 1:Ny, iz in 1:Nz
            x_off = x_sorted[b, t, iy, iz] - xc
            y_off = y_sorted[b, t, iy, iz] - yc
            x_si[b, t, iy, iz] =  x_off * ct + y_off * st + xc
            y_si[b, t, iy, iz] = -x_off * st + y_off * ct + yc
        end
    end

    return TurbineGrid(
        turbine_coordinates, turbine_diameter, wind_directions,
        B, T, xc, yc,
        x_rot, y_rot, z_rot,
        x_sorted, y_sorted, z_sorted,
        sorted_idx, sorted_coord, unsorted_idx,
        fill(radius, T),
        x_si, y_si, z_si,
        grid_resolution,
        nothing,
    )
end

# ── helper: gather along axis 2 ──────────────────────────────────────────────
"""
    _take_along_axis2(src, idx)

Equivalent to `jnp.take_along_axis(src, idx, axis=1)` for a 4-D array where
`idx` holds turbine-order permutations along dimension 2.
"""
function _take_along_axis2(src::Array{Float64,4}, idx::Array{Int,4})
    B, T, Ny, Nz = size(src)
    out = similar(src)
    for b in 1:B, t in 1:T, iy in 1:Ny, iz in 1:Nz
        out[b, t, iy, iz] = src[b, idx[b, t, iy, iz], iy, iz]
    end
    return out
end
