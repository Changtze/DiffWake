# ─── Wake velocity deficit: Gauss model ──────────────────────────────────────

"""
    gaussian_function(C, r, n, sigma)

Gaussian kernel used in the wake velocity deficit model.
"""
gaussian_function(C, r, n, sigma) = C .* exp.(-1 .* r .^ n ./ (2 .* sigma .^ 2))

"""
    safe_sqrt(x; eps=1e-8)

Numerically safe square root (clamps input from below).
"""
safe_sqrt(x; eps::Float64 = 1e-8) = sqrt.(max.(x, eps))

"""
    rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D)

Compute the radial distance `r` and the centreline deficit coefficient `C`
for the Gaussian wake model.
"""
function rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D)
    wv = deg2rad.(wind_veer)

    a = cos.(wv) .^ 2 ./ (2 .* sigma_y .^ 2) .+ sin.(wv) .^ 2 ./ (2 .* sigma_z .^ 2)
    b = -sin.(2 .* wv) ./ (4 .* sigma_y .^ 2) .+ sin.(2 .* wv) ./ (4 .* sigma_z .^ 2)
    c = sin.(wv) .^ 2 ./ (2 .* sigma_y .^ 2) .+ cos.(wv) .^ 2 ./ (2 .* sigma_z .^ 2)

    r = a .* ((y .- y_i .- delta) .^ 2) .- 2 .* b .* (y .- y_i .- delta) .* (z .- HH) .+ c .* ((z .- HH) .^ 2)

    denom = 8.0 .* sigma_y .* sigma_z ./ (D * D)
    safe_denom = ifelse.(denom .< 1e-6, 1e-6, denom)
    d_temp = 1.0 .- (Ct .* cos.(yaw) ./ safe_denom)
    d = clamp.(d_temp, 1e-12, 1.0)
    C = 1.0 .- sqrt.(d)

    return r, C
end

"""
    GaussVelocityDeficit

Gauss velocity-deficit wake model.  Callable struct (functor).
"""
Base.@kwdef struct GaussVelocityDeficit
    alpha::Float64 = 0.58
    beta::Float64  = 0.077
    ka::Float64    = 0.38
    kb::Float64    = 0.004
end

function (m::GaussVelocityDeficit)(
    x_i, y_i, z_i,
    axial_induction_i,
    deflection_field_i,
    yaw_angle_i,
    turbulence_intensity_i,
    ct_i,
    hub_height_i::Float64,
    rotor_diameter_i::Float64;
    x::Array{Float64,4},
    y::Array{Float64,4},
    z::Array{Float64,4},
    u_initial::Array{Float64,4},
    wind_veer::Float64,
)
    # Opposite sign convention
    yaw_angle = -1.0 .* yaw_angle_i
    ct_i = clamp.(ct_i, 1e-6, 0.99999)

    # Initialise velocity deficit
    uR = u_initial .* ct_i ./ (2.0 .* (1.0 .- sqrt.(1.0 .- ct_i)))
    u0 = u_initial .* sqrt.(1.0 .- ct_i)

    # Initial lateral bounds
    sigma_z0 = rotor_diameter_i .* 0.5 .* sqrt.(uR ./ (u_initial .+ u0))
    sigma_y0 = sigma_z0 .* cos.(yaw_angle) .* cos(wind_veer)

    # Bounds of wake regions
    xR = x_i

    x0 = ones(size(u_initial))
    x0 .*= rotor_diameter_i .* cos.(yaw_angle) .* (1.0 .+ sqrt.(1.0 .- ct_i))
    x0 ./= sqrt(2.0) .* (
        4.0 .* m.alpha .* turbulence_intensity_i .+ 2.0 .* m.beta .* (1.0 .- sqrt.(1.0 .- ct_i))
    )
    x0 .+= x_i

    velocity_deficit = zeros(size(u_initial))

    # Masks
    near_wake_mask = (x .> xR .+ 0.1) .* (x .< x0)
    far_wake_mask  = (x .>= x0)

    # ── NEAR WAKE ─────────────────────────────────────────────────────────
    denom_nw = ifelse.(abs.(x0 .- xR) .> 1e-6, x0 .- xR, 1e-6)
    near_wake_ramp_up   = (x .- xR) ./ denom_nw
    near_wake_ramp_down = (x0 .- x) ./ denom_nw

    sigma_y_near  = near_wake_ramp_down .* 0.501 .* rotor_diameter_i .* sqrt.(ct_i ./ 2.0)
    sigma_y_near .+= near_wake_ramp_up .* sigma_y0
    sigma_y_near  = ifelse.(x .>= xR, sigma_y_near, 0.5 .* rotor_diameter_i)

    sigma_z_near  = near_wake_ramp_down .* 0.501 .* rotor_diameter_i .* sqrt.(ct_i ./ 2.0)
    sigma_z_near .+= near_wake_ramp_up .* sigma_z0
    sigma_z_near  = ifelse.(x .>= xR, sigma_z_near, 0.5 .* rotor_diameter_i)

    r_near, C_near = rC(
        wind_veer, sigma_y_near, sigma_z_near,
        y, y_i, deflection_field_i, z,
        hub_height_i, ct_i, yaw_angle, rotor_diameter_i,
    )

    near_wake_deficit  = gaussian_function(C_near, r_near, 1, sqrt(0.5))
    near_wake_deficit .*= near_wake_mask
    velocity_deficit  .+= near_wake_deficit

    # ── FAR WAKE ──────────────────────────────────────────────────────────
    ky = m.ka .* turbulence_intensity_i .+ m.kb
    kz = m.ka .* turbulence_intensity_i .+ m.kb
    sigma_y_far = (ky .* (x .- x0) .+ sigma_y0) .* far_wake_mask .+ sigma_y0 .* (x .< x0)
    sigma_z_far = (kz .* (x .- x0) .+ sigma_z0) .* far_wake_mask .+ sigma_z0 .* (x .< x0)

    r_far, C_far = rC(
        wind_veer, sigma_y_far, sigma_z_far,
        y, y_i, deflection_field_i, z,
        hub_height_i, ct_i, yaw_angle, rotor_diameter_i,
    )

    far_wake_deficit  = gaussian_function(C_far, r_far, 1, sqrt(0.5))
    far_wake_deficit .*= far_wake_mask
    velocity_deficit .+= far_wake_deficit

    return velocity_deficit, nothing
end
