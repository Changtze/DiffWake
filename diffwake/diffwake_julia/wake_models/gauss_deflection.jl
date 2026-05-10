# ─── Wake deflection: Gauss model + GCH transverse velocities ────────────────

"""
    GaussVelocityDeflection

Gaussian wake deflection model.  Callable struct (functor).
"""
Base.@kwdef struct GaussVelocityDeflection
    ad::Float64   = 0.0
    bd::Float64   = 0.0
    alpha::Float64 = 0.58
    beta::Float64  = 0.077
    ka::Float64    = 0.38
    kb::Float64    = 0.004
    dm::Float64    = 1.0
    eps_gain::Float64 = 0.2
    use_secondary_steering::Bool = true
end

function (m::GaussVelocityDeflection)(
    x_i,        # (B,1,1,1)
    yaw_i,      # (B,1,1,1)
    turb_I_i,   # (B,1,Ny,Nz)
    ct_i,       # (B,1,1,1)
    D::Float64,
    x::Array{Float64,4},
    U_free::Array{Float64,4},
    wind_veer::Float64,
)
    sqrt2 = sqrt(2.0)
    veer  = deg2rad(wind_veer)

    # Opposite sign convention
    yaw   = -1.0 .* yaw_i
    cos_y = cos.(yaw)
    one_  = ones(size(cos_y))

    uR = U_free .* ct_i .* cos_y ./ (2.0 .* (one_ .- sqrt.(one_ .- ct_i .* cos_y)))
    u0 = U_free .* sqrt.(one_ .- ct_i)

    denom = sqrt2 .* (4.0 .* m.alpha .* turb_I_i .+ 2.0 .* m.beta .* (one_ .- sqrt.(one_ .- ct_i)))
    x0 = D .* cos_y .* (1.0 .+ sqrt.(one_ .- ct_i .* cos_y)) ./ denom .+ x_i

    k = m.ka .* turb_I_i .+ m.kb

    sigma_z0 = D .* 0.5 .* sqrt.(uR ./ (U_free .+ u0))
    sigma_y0 = sigma_z0 .* cos_y .* cos(veer)

    theta_c0 = m.dm .* 0.3 .* yaw ./ cos_y
    theta_c0 .*= (1.0 .- sqrt.(one_ .- ct_i .* cos_y))
    delta0 = tan.(theta_c0) .* (x0 .- x_i)

    mask_near = (x .>= x_i) .& (x .<= x0)
    mask_far  = x .> x0

    # ── Near-wake deflection ──────────────────────────────────────
    delta_near = ((x .- x_i) ./ (x0 .- x_i)) .* delta0 .+ (m.ad .+ m.bd .* (x .- x_i))
    delta_near = ifelse.(mask_near, delta_near, zeros(size(delta_near)))

    # ── Far-wake deflection ───────────────────────────────────────
    sigma_y = ifelse.(mask_far, k .* (x .- x0) .+ sigma_y0, sigma_y0)
    sigma_z = ifelse.(mask_far, k .* (x .- x0) .+ sigma_z0, sigma_z0)

    C0   = 1.0 .- u0 ./ U_free
    M0   = C0 .* (2.0 .- C0)
    E0   = C0 .^ 2 .- 3.0 .* exp(1 / 12) .* C0 .+ 3.0 .* exp(1 / 3)
    M0_s = sqrt.(M0)

    mid    = sqrt.(sigma_y .* sigma_z ./ (sigma_y0 .* sigma_z0))
    ln_num = (1.6 .+ M0_s) .* (1.6 .* mid .- M0_s)
    ln_den = (1.6 .- M0_s) .* (1.6 .* mid .+ M0_s)
    log_t  = log.(ln_num ./ ln_den)

    mult = theta_c0 .* E0 ./ 5.2 .* sqrt.(
        sigma_y0 .* sigma_z0 ./ (k .* k .* M0)
    )

    delta_far = delta0 .+ mult .* log_t .+ (m.ad .+ m.bd .* (x .- x_i))
    delta_far = ifelse.(mask_far, delta_far, zeros(size(delta_far)))

    return delta_near .+ delta_far
end


# ─── Transverse velocity helpers ─────────────────────────────────────────────

const NUM_EPS   = 0.001
const NUM_EPS_2 = 1e-7
const EPS_GAIN  = 0.2

"""
    _gamma(D, velocity, Uinf, Ct, scale)

Circulation parameter for vortex model.
"""
_gamma(D, velocity, Uinf, Ct, scale) = scale .* (π / 8.0) .* D .* velocity .* Uinf .* Ct

"""
    _cbrt(x)

Signed cube root.
"""
_cbrt(x) = sign.(x) .* abs.(x) .^ (1.0 / 3.0)


"""
    calculate_transverse_velocity(...)

Compute vortex-based transverse (V) and vertical (W) wake velocities.
"""
function calculate_transverse_velocity(
    u_i::Array{Float64,4},            # (B,1,Ny,Nz)
    u_initial::Array{Float64,4},
    dudz_initial::Array{Float64,4},
    delta_x::Array{Float64,4},
    delta_y::Array{Float64,4},
    z_coord::Array{Float64,4},
    rotor_diameter::Float64,
    hub_height::Float64,
    yaw::Array{Float64},              # (B,1,1,1)
    ct_i::Array{Float64},
    tsr_i,
    axial_induction_i::Array{Float64},
    wind_shear::Float64;
    scale::Float64 = 1.0,
)
    D   = rotor_diameter
    HH  = hub_height
    Ct  = ct_i
    aI  = axial_induction_i

    # Mean free-stream (B,1,1,1)
    B = size(u_initial, 1)
    Uinf = reshape([mean(u_initial[b, :, :, :]) for b in 1:B], B, 1, 1, 1)
    eps_val = EPS_GAIN * D
    ones_ = ones(size(Ct))

    s_c = sin.(yaw) .* cos.(yaw)
    vel_top    = ((HH + 0.5 * D) / HH) ^ wind_shear .* ones_
    vel_bottom = ((HH - 0.5 * D) / HH) ^ wind_shear .* ones_

    Gamma_top    =  s_c .* _gamma(D, vel_top,    Uinf, Ct, scale)
    Gamma_bottom = -s_c .* _gamma(D, vel_bottom, Uinf, Ct, scale)

    # Turbine average velocity (cube root mean)
    mean_cubed = zeros(B, 1, 1, 1)
    Ny, Nz = size(u_i, 3), size(u_i, 4)
    for b in 1:B
        mean_cubed[b, 1, 1, 1] = mean(u_i[b, 1, :, :] .^ 3)
    end
    turbine_avg_u = _cbrt(mean_cubed)
    Gamma_core = 0.5 .* π .* D .* (aI .- aI .^ 2) .* turbine_avg_u ./ tsr_i

    # Eddy viscosity
    lmda  = D / 8.0
    kappa = 0.41
    lm = kappa .* z_coord ./ (1.0 .+ kappa .* z_coord ./ lmda)
    nu = lm .^ 2 .* abs.(dudz_initial)

    decay = eps_val^2 ./ (4.0 .* nu .* delta_x ./ Uinf .+ eps_val^2)
    y_loc = delta_y .+ NUM_EPS

    function vortex(Gamma, z_shift)
        z_ = z_coord .- z_shift .+ NUM_EPS
        r2 = y_loc .^ 2 .+ z_ .^ 2
        core = 1.0 .- exp.(-r2 ./ eps_val^2)
        V =  (Gamma .* z_) ./ (2.0 .* π .* r2) .* core .* decay
        W = (-Gamma .* y_loc) ./ (2.0 .* π .* r2) .* core .* decay
        return V, W
    end

    V1, W1 = vortex( Gamma_top,            HH + 0.5 * D)
    V2, W2 = vortex( Gamma_bottom,         HH - 0.5 * D)
    V5, W5 = vortex( Gamma_core,           HH)

    V3, W3 = vortex(-Gamma_top,          -(HH + 0.5 * D))
    V4, W4 = vortex(-Gamma_bottom,       -(HH - 0.5 * D))
    V6, W6 = vortex(-Gamma_core,         -HH)

    V = V1 .+ V2 .+ V3 .+ V4 .+ V5 .+ V6
    W = W1 .+ W2 .+ W3 .+ W4 .+ W5 .+ W6

    V = ifelse.(delta_x .+ NUM_EPS_2 .>= 0.0, V, zeros(size(V)))
    W = ifelse.((delta_x .+ NUM_EPS_2 .>= 0.0) .& (W .+ NUM_EPS_2 .>= 0.0), W, zeros(size(W)))

    return V, W
end


"""
    yaw_added_turbulence_mixing(u_i, I_i, v_i, w_i, turb_v_i, turb_w_i)

GCH yaw-added turbulence mixing contribution.
"""
function yaw_added_turbulence_mixing(u_i, I_i, v_i, w_i, turb_v_i, turb_w_i)
    B = size(u_i, 1)

    # I_i: (B,1,Ny,Nz) → take [:,1,1,1]
    I_vec = [I_i[b, 1, 1, 1] for b in 1:B]

    average_u_i = [sign(u_i[b,1,iy,iz]) * abs(u_i[b,1,iy,iz])^3
                   for b in 1:B, iy in 1:size(u_i,3), iz in 1:size(u_i,4)]
    average_u = [mean(average_u_i[b, :, :])^(1/3) for b in 1:B]

    k = (average_u .* I_vec) .^ 2 ./ (2.0 / 3.0)
    u_term = sqrt.(2.0 .* k)
    v_term = [mean(v_i[b,1,:,:] .+ turb_v_i[b,1,:,:]) for b in 1:B]
    w_term = [mean(w_i[b,1,:,:] .+ turb_w_i[b,1,:,:]) for b in 1:B]

    k_total = 0.5 .* (u_term .^ 2 .+ v_term .^ 2 .+ w_term .^ 2)
    I_total = sqrt.((2.0 / 3.0) .* k_total) ./ average_u

    I_mixing = I_total .- I_vec

    return reshape(I_mixing, B, 1, 1, 1)
end


"""
    wake_added_yaw(...)

Compute secondary-steering added yaw from transverse velocities.
"""
function wake_added_yaw(
    u_i, v_i, u_initial, delta_y, z_i,
    rotor_diameter, hub_height, ct_i,
    tip_speed_ratio, axial_induction_i, wind_shear;
    scale::Float64 = 1.0,
)
    D  = rotor_diameter
    HH = hub_height
    Ct = ct_i
    TSR = tip_speed_ratio
    aI  = axial_induction_i

    B = size(u_i, 1)
    Ny, Nz = size(u_i, 3), size(u_i, 4)

    avg_v = zeros(B, 1)
    for b in 1:B
        avg_v[b, 1] = mean(v_i[b, 1, :, :])
    end

    Uinf = reshape([mean(u_initial[b, :, :, :]) for b in 1:B], B, 1, 1, 1)
    eps_val = 0.2 * D
    ones_ = ones(size(Ct))

    vel_top    = ((HH + D / 2) / HH) ^ wind_shear .* ones_
    Gamma_top  = _gamma(D, vel_top, Uinf, Ct, scale)

    vel_bottom    = ((HH - D / 2) / HH) ^ wind_shear .* ones_
    Gamma_bottom  = -_gamma(D, vel_bottom, Uinf, Ct, scale)

    mean_cubed = zeros(B, 1, 1, 1)
    for b in 1:B
        mean_cubed[b, 1, 1, 1] = mean(u_i[b, 1, :, :] .^ 3)
    end
    turbine_average_velocity = sign.(mean_cubed) .* abs.(mean_cubed) .^ (1/3)

    Gamma_wake_rotation = 0.25 .* 2 .* π .* D .* (aI .- aI .^ 2) .* turbine_average_velocity ./ TSR

    yLocs = delta_y .+ NUM_EPS

    function vortex_velocity(Gamma, z_shift)
        z_ = z_i .- z_shift .+ NUM_EPS
        r_ = yLocs .^ 2 .+ z_ .^ 2
        core_shape = 1.0 .- exp.(-r_ ./ eps_val^2)
        return (Gamma .* z_) ./ (2.0 .* π .* r_) .* core_shape
    end

    v_top    = reshape([mean(vortex_velocity(Gamma_top,            HH + D / 2)[b,1,:,:]) for b in 1:B], B, 1)
    v_bottom = reshape([mean(vortex_velocity(Gamma_bottom,         HH - D / 2)[b,1,:,:]) for b in 1:B], B, 1)
    v_core   = reshape([mean(vortex_velocity(Gamma_wake_rotation,  HH)[b,1,:,:])         for b in 1:B], B, 1)

    val = 2.0 .* (avg_v .- v_core) ./ (v_top .+ v_bottom)
    val = clamp.(val, -0.99999999, 0.99999999)
    y_out = 0.5 .* asin.(val)    # radians

    return reshape(y_out, B, 1, 1, 1)
end
