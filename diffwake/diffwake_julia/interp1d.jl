"""
    interp1d(x, y, xnew)

1-D linear interpolation with clamping outside the data range.

`x` must be sorted in ascending order.  Values of `xnew` below `x[1]` return
`y[1]`; values above `x[end]` return `y[end]`.
"""
function interp1d(x::AbstractVector, y::AbstractVector, xnew::AbstractArray)
    # searchsortedlast gives the index of the last element <= xnew[k]
    indices = map(v -> searchsortedlast(x, v), xnew)
    indices = clamp.(indices, 1, length(x) - 1)

    x0 = x[indices]
    x1 = x[indices .+ 1]
    y0 = y[indices]
    y1 = y[indices .+ 1]

    slope = (y1 .- y0) ./ (x1 .- x0 .+ 1e-16)
    ynew  = y0 .+ slope .* (xnew .- x0)

    # Clamp outside range
    ynew = ifelse.(xnew .< x[1],   y[1],   ynew)
    ynew = ifelse.(xnew .> x[end], y[end], ynew)

    return ynew
end

# Scalar convenience
function interp1d(x::AbstractVector, y::AbstractVector, xnew::Real)
    idx = searchsortedlast(x, xnew)
    idx = clamp(idx, 1, length(x) - 1)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    slope = (y1 - y0) / (x1 - x0 + 1e-16)
    ynew  = y0 + slope * (xnew - x0)
    xnew < x[1]   && return y[1]
    xnew > x[end]  && return y[end]
    return ynew
end
