# ─── Wake turbulence: Crespo-Hernández model ─────────────────────────────────

"""
    CrespoHernandez

Wake-added turbulence intensity model (Crespo & Hernández).
"""
Base.@kwdef struct CrespoHernandez
    initial::Float64    = 0.1
    constant::Float64   = 0.9
    ai::Float64         = 0.8
    downstream::Float64 = -0.32
end

function (m::CrespoHernandez)(
    ambient_TI::Array{Float64},       # (B,1,1,1)
    x::Array{Float64,4},              # (B,T,Ny,Nz)
    x_i::Array{Float64},              # (B,1,1,1)
    rotor_diameter::Float64,
    axial_induction::Array{Float64},  # (B,1,1,1)
)
    delta_x = x .- x_i

    # Avoid zero / negative downstream distance
    delta_x_safe = ifelse.(delta_x .> 0.1, delta_x, ones(size(delta_x)))

    ti_add = (
        m.constant .*
        axial_induction .^ m.ai .*
        ambient_TI .^ m.initial .*
        (delta_x_safe ./ rotor_diameter) .^ m.downstream
    )

    # Zero out upstream contributions
    ti_add = ti_add .* Float64.(delta_x .> -0.1)

    # Sanitise
    ti_add = replace(ti_add, NaN => 0.0, Inf => 0.0)

    return ti_add
end
