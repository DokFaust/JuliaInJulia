NUM_MAXITER = 1000

function get_pixel_color(fracgen::FractalGenerator,px::Int64,py::Int64)
    r=0.0
    g=0.0
    b=0.0

    for i in 1:fracgen.imagespec.num_supersamples

        dx = Float64(i-1)/fracgen.imagespec.num_supersamples
        x  = ( (Float64(px) + dx) / fracgen.imagespec.height - 0.5) * fracgen.cameraspec.zoom +
                fracgen.cameraspec.center.re

        for j in 1:fracgen.imagespec.num_supersamples

            dy = Float64(j-1) / fracgen.imagespec.num_supersamples
            y  = ((Float64(py) + dy) / fracgen.imagespec.width -0.5) * fracgen.cameraspec.zoom +
                fracgen.cameraspec.center.im
            prop = get_num_iters(fracgen.fracspec.fracfunc_closure(Complex(x,y))...)

            if fracgen.cameraspec.affine
                prop = max(0.0, prop)
            else
                prop = max(0.0, prop/NUM_MAXITER)
            end

            prop = prop^fracgen.cameraspec.pow
            color = blend(fracgen.cameraspec.gradient,prop)

            r+=color.r
            g+=color.g
            b+=color.b
        end #1st loop
    end#2nd ith loop

    nss = fracgen.imagespec.num_supersamples*fracgen.imagespec.num_supersamples

    r/=nss
    g/=nss
    b/=nss

    return RGB8bit(r,g,b)

end

function get_num_iters(f::Function, zinit::Complex=zero(Complex))

    #I is the numb of iters before RUNAWAY or goin over THRESHOLD,
    #zinit represents the starting to calc julai SET
    i=0
    THRESHOLD = 2^16
    z=zinit

    #Std mandelbrot,julai update rule
    #COMPUTATIONALLY EXPENSIVE
    while i < NUM_MAXITER
        z=f(z)
        if real(z*z') >= THRESHOLD
            break
        end
        i += 1
    end

    if i == NUM_MAXITER
        return 0.0
    end

    #Both considering the runaway iter and the last point computed
    i = Float64(i)
    i += (1.0-log2(log2(real(z*z'))/2.0))
    return i
end

function blend(gradient::Gradient, prop::Float64)

    if prop == 0.0
        return gradient.nicolor
    end

    return blend(gradient.startcolor, gradient.endcolor, prop)

end


function blend(startcolor::RGB8bit, endcolor::RGB8bit, prop::RGB8bit)

    bl = (s,e,p)->p*s+e*(1.0-p) #blend between s,e

    function blnd(s,e,p)
        #have color values 1.0 wrap around
        o = bl(s,e,p) % 1.0
        return o > 0 ? o : 1.0+o
    end

    return RGB8bit(
        blnd(startcolor.r, endcolor.r, prop),
        blnd(startcolor.g, endcolor.g, prop),
        blnd(startcolor.b, endcolor.b, prop)
    )

end
