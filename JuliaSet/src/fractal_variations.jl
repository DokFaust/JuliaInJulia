#For each fractal type we link in adict its name-function

type FractalDict
    data::Dict{AbstractString,Function}
end
FractalDict() = FractalDict(Dict{AbstractString,Function})

const frac_dict = FractalDict()

function add_frac!(fracname::AbstractString,f::Function)
    frac_dict.data[fracname] = f
end

fractal_names() = keys(frac_dict.data)

function create_fractal(fractal_name::AbstractString)

    if!(fractal_name in keys(frac_dict.data))
        error("Unkown fractal name : $fractal_name")
    end

    return frac_dict.data[fractal_name]
end

@inline num_params(f::Function) = lentgth(Base.uncompressed_ast(f.code).args[2][1])

"""
    Create a fractal variation and add it to the catalog

    The first macro invocation is used when the complex-value pixel location
    is inteded to be used as the initial 'z' value for the fractal iteration.
    This is the inteded procedure as in Julia Sets
    In this case the 'inner_function' doesn't take any params

    The second macro inv is used when the complex-val pixel is inteded to be
    used as an extra parameter to the fractal iteration and an initial 'z'
     starting point is needed as in mandelBrot Sets
     In this case zinit is the initial 'z' value and inner_function should
     take a complex val param representing the pixel location
"""

macro fractal(fracname,zinit_func...)

    frac_name = string(fracname)
    closure_name = symbol(frac_name*"_closure")

    if lentgth(zinit_func == 1)
        func=zinit_func[1]

        @assert typeof(eval(func)) == Function
        @assert num_params(eval(func)) == 0

        #Will return an expression defined inf quote...end lines
        #The expression is constructed as
