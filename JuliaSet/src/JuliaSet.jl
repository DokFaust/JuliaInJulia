module juliaset

export Gradient, ImageSpec, CameraSpec, FractalSpec, JuliaSetGenerator

include("types.jl")
include("parse.jl")
export fractal_names, create_fractal
include("fractal_variations.jl")
export generate_juliaset
include("juliaset_generate.jl")
include("core_set.jl")

end
