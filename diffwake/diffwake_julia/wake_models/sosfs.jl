# ─── Wake combination: Sum-of-Squares Free-Stream Superposition ──────────────

"""
    SOSFS

Combines wake deficits via sum-of-squares (hypot).
"""
struct SOSFS end

function (m::SOSFS)(wake_field::Array{Float64}, velocity_field::Array{Float64})
    return hypot.(wake_field, velocity_field), nothing
end
