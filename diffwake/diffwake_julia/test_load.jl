wpush!(LOAD_PATH, raw"E:\Aeronautics\fyp\DiffWake\diffwake\diffwake_julia")
include(raw"E:\Aeronautics\fyp\DiffWake\diffwake\diffwake_julia\DiffWakeJulia.jl")
using .DiffWakeJulia
println("Module loaded successfully")
println("Exported names: ", names(DiffWakeJulia))
