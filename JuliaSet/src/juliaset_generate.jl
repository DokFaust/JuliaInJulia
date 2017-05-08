using Images
using Iterators

type FractalGenerator
    cameraspec::CameraSpec
    images::ImageSpec
    fracspec::FractalSpec
end

function Base.call(fracgen::FractalGenerator)
    f = "temp.png"

    @time img = generate_juliaset(fracgen)
    return ("Draw a wonderful fractal! $(now())", f)
end

function generate_juliaset(fracgen::FractalGenerator)

    w,h = fracgen.imagespec.width, fracgen.imagespec.height
    simgdata = SharedArray(RGB8bit,(w,h))

    #Using a shared matrix of type RGB8bit and updating
    #it using synchrouns parallel computation of the
    #master processes that share the img data

    @sync @parallel for j in 1:w
        @inbounds @fastmath for i in 1:h
            simgdata[i,j] = get_pixel_color(fracgen,j,i)
        end
    end

    #Ret the actual Array object backing simgdata
    imagedata = Array{RGB8bit}(w,h)
    imagedata[:,:] = sdata(simgdata)

    return Image(imagedata)
end
