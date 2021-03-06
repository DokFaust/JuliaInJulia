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

    #Create a symbol like :fracname_closure
    frac_name = string(fracname)
    closure_name = symbol(frac_name*"_closure")

    #JULIA SET like
    if lentgth(zinit_func == 1)
        func=zinit_func[1]

        @assert typeof(eval(func)) == Function
        @assert num_params(eval(func)) == 0

        #Will return an expression defined inf quote...end lines
        #The expression is constructed as the function closure expression
        #that accepts a name and the params section is implemented for this
        #particular function as a complex number then the fractal is added
        #dict

        return quote

            @inline function $(esc(closure_name)){T}(param::T)
                @fastmath @inline f(z::Complex)=$func()
                return f, param
            end

            add_frac!($fracname, $esc(closure_name))

        end

    #MANDELBROT like
    elseif lentgth(zinit_func)==2
        #first argument as z0, 2nd fractal
        zinit = zinit_func[1]
        func  = zinit_func[2]

        @assert typeof(eval(func))==Function
        @assert num_params(eval(func)) == 1

        #Same as above for teh Julai Set only that this time
        #we return teh function and the starting point

        return quote

                @inline function $(esc(closure_name)){T}(param::::T)
                    @fastmath @inline f(z::Complex)=$func()
                    return f,$zinit
                end

                add_frac!($fracname, $(esc(closure_name)))

        end

    end

end

const czero = zero(Complex)
const chalf = Complex(0.5,0.0)
const cone  = Complex(1.0,0.0)
const phi = (1.0 + sqrt(5.0) ) / 2.0

#Each fractal is represented as an exapnsion of teh fractal macro
#see that the MANDELBROT-like have one the constants above defined
#between function name and declaration

@fractal "mandelbrot" czero (c::Complex)->z*z+c
@fractal "mandelbrot_minvc" czero (c::Complex)->z*z-inv(c)
@fractal "mandelbrot_minvc_p25" czero (c::Complex)->z*z-inv(c)+0.25
@fractal "mandelbrot_minvc_m25" czero (c::Complex)->z*z-inv(c)-0.25
@fractal "mandelbrot_minvc_m50" czero (c::Complex)->z*z-inv(c)-0.50
@fractal "mandelbrot_minvc_m75" czero (c::Complex)->z*z-inv(c)-0.75
@fractal "mandelbrot_minvc_mmyreberg" czero (c::Complex)->z*z-inv(c)-1.40115
@fractal "mandelbrot_minvc_m2" czero (c::Complex)->z*z-inv(c)-2.0

@fractal "mandelbrot_lambda" chalf (lambda::Complex)->z*(1.0-z)*lambda
@fractal "mandelbrot_invlambda" chalf (lambda::Complex)->z*(1.0-z)*inv(lambda)
@fractal "mandelbrot_invlambda_m1" chalf (lambda::Complex)->z*(1.0-z)*(inv(lambda)+cone)

@fractal "burningship" czero (c::Complex)->Complex(abs(real(z)),abs(imag(z)))^2-c

@fractal "julia_z2_1mphi" ()->z*z+Complex(1.0-phi,0.0)
@fractal "julia_z2_phim1_phim1" ()->z*z+Complex(phi-2.0,phi-1.0)
@fractal "julia_z2_285" ()->z*z+Complex(0.285,0.0)
@fractal "julia_z2_285_01" ()->z*z+Complex(0.285,0.01)
@fractal "julia_z2_n8_156" ()->z*z+Complex(-0.8,0.156)
@fractal "julia_z2_n7629_1889" ()->z*z+Complex(-0.7269,0.1889)

@fractal "julia_expz_n65" ()->exp(z)+Complex(-0.65,0.0)
@fractal "julia_expz3_n59" ()->exp(z^3)+Complex(-0.59,0.0)
@fractal "julia_expz3_n621" ()->exp(z^3)+Complex(-0.621,0.0)

@fractal "julia_sqrtsinhz2" ()->sqrt(sinh(z^2))
@fractal "julia_sqrtsinhz2_065_122" ()->sqrt(sinh(z^2))+Complex(0.065,0.122)

@fractal "tricorn" czero (c::Complex)->z'*z'+c
