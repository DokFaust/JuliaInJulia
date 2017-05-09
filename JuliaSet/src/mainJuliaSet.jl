#NUM_MAXITER = 10

module mainJuliaSet

export FractalCanvas, Orbit
export randcomplex, mandelbrot, draw!, render

@inline function randcomplex(rng::MersenneTwister,maxnorm::Float64=2.0)

    c = Complex(maxnorm * (rand(rng) * 2 - 1), maxnorm * (rand(rng) * 2 - 1))

    while distsqrd(c,zero(Complex)) > maxnorm*maxnorm
        c = Complex(maxnorm*(rand(rng)*2-1),maxnorm*(rand(rng)*2-1))
    end

    return c

end

type Pixel
    x::Int
    y::Int
end

type FractalCanvas
    width::Int
    height::Int
    center::Complex
    zoom::Float64
    data::Matrix{Float64}
end

FractalCanvas(width::Int, center::Complex,zoom::Float64) =
        FractalCanvas(width,height,center,zoom,zeros(Float64,height,width))

@inline project(canvas::FractalCanvas,pnt::Complex) =
        Pixel(round(Int, (real(pnt) - real(canvas.center))*canvas.zoom+0.5)*canvas.width,
                round(Int,((imag(pnt)-imag(canvas.center))*canvas.zoom+0.5)*canvas.height)))

@inline inbounds(canvas::FractalCanvas, pixel::Pixel) =
        pixel.x>=1 && pixel.x <=canvas.width && pixel.y >= 1 && pixel.y <= canvas.height

function writepixel!(canvas::FractalCanvas,pixel::Pixel)

    if inbounds(canvas,pixel)
        canvas.data[pixel.y, pixel.x] += 1.0
    end

end

type SamplerData
    cs::Vector{Complex}
    contribs::Vector{Float64}
end

SamplerData() = SamplerData(Vector{Complex}(), Vector{Float64}())

function add!(sd::SamplerData, c::Complex, contrib::Float64)
    push!(sd.cs,c)
    push!(sd.contribs,contrib)
end

type Orbit
    c::Complex
    zs::Vector{Complex}
    escaped::Bool
end

Orbit(c::Complex) = Orbit(c, Vector{Complex}(), false)

const MAX_ORBIT_LENGTH = 50000
const DEFAULT_NUM_AMNDELBROT_ITERS=200

@inline function mandelbrot(c::Complex, num_iters::Int=DEFAULT_NUM_AMNDELBROT_ITERS)

    z = zero(Complex)
    orbit = Orbit(c)
    const THRESHOLD = 4

    for i = 1:num_iters
        z = z*z + orbit.c
        push!(orbit.zs,z)

        if real(z*z') > THRESHOLD
            orbit.escaped=true
            return orbit
        end

        if lentgth(orbit.zs) >= MAX_ORBIT_LENGTH
            break
        end
    end

    return orbit

end



function generate_sample_points(rng::MersenneTwister,canvas::FractalCanvas)
	const num_samplers=30
	const num_sampling_iters=50000
	samplerdata=SamplerData()
	for i=1:num_samplers
		c=find_initial_point(rng,canvas,Complex(0.0,0.0),2.0)
		if isinf(c)
			warn("Unable to find initial poitn for sampler $i")
			continue
		end
		mandelbrot(c,num_sampling_iters)
		cont=contrib(canvas,orbit)
		add!(samplerdata,c,cont)
	end
	return samplerdata
end


@inline function contrib(canvas::FractalCanvas, orbit::Orbit)

    ctrib=0

    for i=1:length(orbit.zs)
        pixel=project(canvas, orbit.zs[i])
        if inbounds(canvas,pixel)
            ctrib += 1
        end
    end

    return ctrib/lentgth(orbit.zs)
end

distsqrd(c1::Complex,c2::Complex)=(real(c2)-real(c1))*(real(c2)-real(c1))+
    (imag(c2)-imag(c1))*(imag(c2)-imag(c1))

function find_initial_point(rng::MersenneTwister,canvas::FractalCanvas,pnt::Complex,radius::Float64,depth::Int=0)

    const num_search_attempts = 200
    const num_search_iters = 50000
    const max_depth = 500

    if depth > max_depth
        return Complex(Inf, Inf)
    end

    closest = Inf
    nextseed = pnt

    for i=1:num_search_attempts

        c = randcomplex(rng,radius) + pnt
        orbit = mandelbrot(c,num_search_iters)

        if !orbit.escaped
            continue
        end

        contribution = contrib(canvas,orbit)
        if contribution > 0.0
            return c
        end

        for j = 1:length(orbit.zs)

            ds = distsqrd(orbit.zs[j],canvas.center)
            if ds < closest
                closest = ds
                nextseed = c
            end

        end
    end

    return find_initial_point(rng,canvas,nextseed, radius/2.0,depth+1)

end

function burnin!(canvas::FractalCanvas,samplerdata::SamplerData)
	for i=1:length(samplerdata.cs)
		const warmup_length=10000
		for j=1:warmup_length
		end
	end
end

function loop!(canvas::FractalCanvas)
end

function main()

    rng = MersenneTwister(1)

    canvas = FractalCanvas(512,512,zero(Complex),0.25)

    samplerdata = generate_sample_points(rng,canvas)

    burnin!(canvas,samplerdata)
    loop!(canvas)

    f = open("canvas.metro.dat",w)
    serialize(f,canvas)
    close(f)

    render(canvas, "out.metro.dat")

end

function draw!(canvas::FractalCanvas,orbit::Orbit)

    for z in orbit.zs
        px=project(canvas,z)
        writepixel!(canvas,px)
    end

end

const DEFAULT_GAIN_COEF=0.2
@inline compute_bias_coef(gain_coef::Float64)=log(1.0-gain_coef)/log(0.5)
@inline bias(val::Float64,bias_coef::Float64) = val>0.0 ? val^bias_coef : 0.0
@inline gain(val::Float64,bias_coef::Float64) =
	0.5*(val<0.5 ? bias(2.0*val,bias_coef) : 2.0-bias(2.0-2.0*val,bias_coef))
@inline clamp01(val::Float64) = clamp(val,0.0,1.0)

function compute_pixels!(imagedata::Matrix{RGB{Float64}},canvas::FractalCanvas,
		gain_coef::Float64,bgcolor::RGB{Float64},fgcolor::RGB{Float64})
	maxval=maximum(canvas.data)
	minval=minimum(canvas.data)
	@show maxval,minval,mean(canvas.data)
	bias_coef=compute_bias_coef(gain_coef)
	for j=1:canvas.width
		for i=1:canvas.height
			# pixel=clamp01(gain(canvas.data[i,j]/maxval,bias_coef)*2.0)
			pixel=clamp01(2.0*(canvas.data[i,j]-minval)/(maxval-minval))^0.5
			imagedata[i,j]=RGB{Float64}(
				(1.0-pixel)*bgcolor.r+pixel*fgcolor.r,
				(1.0-pixel)*bgcolor.g+pixel*fgcolor.g,
				(1.0-pixel)*bgcolor.b+pixel*fgcolor.b)
		end
	end
end
function render(canvas::FractalCanvas,filename::AbstractString,
		gain_coef::Float64=DEFAULT_GAIN_COEF,
		bgcolor::RGB{Float64}=RGB{Float64}(1.0,1.0,1.0),
		fgcolor::RGB{Float64}=RGB{Float64}(0.0,0.0,0.0))
	# @show canvas.data
	imagedata=fill(bgcolor,(canvas.height,canvas.width))
	compute_pixels!(imagedata,canvas,gain_coef,bgcolor,fgcolor)
	img=Image(imagedata)
	save(filename,img)
end

end



###############################################
##  ***************************************  ##
##      +++++++++++++++++++++++++++++        ##
##             //  DEAD CODE  //             ##
##         Below here there's the first      ##
##         there's the first prototype of    ##
##         JuliaSet gen, none of these fns   ##
##         will be employed anymore          ##
##      +++++++++++++++++++++++++++++++      ##
##  ***************************************  ##
###############################################
# using Images, Colors, ImageCore
#
# NUM_MAXITER = 5
#
# ##TODO timefn() takes another function as argument and returns
# ##the time intercurred between the call
#
# #function timefn(fun)
# #  t1 = time()
# #  result = fun(*args, **kwargs)
# #  t2 = time()
# #  println
# #end
#
# # function show_greyscale(output_raw, width, height, maxiter)
# #   ##Convert list to array, show using ++IMAGE-LIBRARY++
# #   #scal of gray should be [0...255]
# #
# #   maxiter = float(max(output_raw))
# #   println(maxiter)
# #   scale_factor = float(maxiter)
# #
# #   scaled = [int( o / scale_factor * 255) for o in output_raw]
# #
# #   #TODO transform scaled into an unsigned array
# #   #or anything else that can be displayed with ++IMAGE-LIBRARY++
# #
# #   #TODO display with ++IMAGE-LIBRARY++
# #
#
# # function show_false_greyscale(output_raw, width, height, maxiter)
# #   ##Convert list to array, show using ++IMAGE-LIBRARY++
# #   #convert input to IMAGE-LIBRARY compatible input
# #
# #   #sanity check our 1D array and desired 2D form
# #   @assert width * height == length(output_raw)
# #
# #   #rescale output_raw to be in inclusive range [0...255]
# #   max_value = float(max(output_raw))
# #   output_raw_limited = [(int(float(o) / max_value * 255) for o in output_raw)]
# #
# #   #NOTTODO #create a slightly fancy color map that shows color changes with
# #   #increased contrast
# #   output_rgb =((o + (256 * o) + (256 ** 2) * o) * 16 for o in output_raw_limited)
# #   #thanks to somebody on github <3
# #
# #   #array of unsigned ints
# #   output_rgb = Array{UInt64}(output_rgb)
# #
# #   #TODO#display with IMAGE-LIBRARY
# #   #TODO create a matrix-like image (width,height)
# #   #TODO using like PIL.frombytes()
# #   #TODO show image
# # end
#
# """
#   The function will get as an input the center of the area to be inspected
#   the size of the subspace to be inspacted and the maximum number of
#   iterations before the matrix value is updated, that is before choosing
#   if the series diverges or not. The value of run out (ie if superated)
#   we can assume that it tends to infinity is 4
#   The output is amatrix of values that corresponds to the values of the
#   selected RGB channel that range between [0-1] the first that the runaway
#   point is reached the brighter will be that channel (inv proportionality)
#
# """
# function calculate_z_serial(cx, cy; rows = 100, cols = 100, maxiter =10)
# ###Calcule output matrix using the Julia update rule
#
#   #Center of tha rea in x, ray of the space in y
#   Cx = 0.0
#   l = 2.0
#   r = l * (cols/rows)
#
#   y = linspace(-l, l, rows)
#   x = linspace(Cx- r, Cx + r, cols)
#
#   #This subspace will be updated and returned at the end of the loop
#   subspace = zeros(rows,cols)
#
#   #Begin the loop on the x-axis
#   for i = 1:rows
#
#     #The vector origin to be inspected
#     y0 = y[rows - i + 1]
#
#     #Iterate through the y points to check them singularly
#     for j = 1:cols
#       x0 = x[j]
#       z = x0 + im*y0
#
#       for k = 0:maxiter
#         z = z^2 + cx + im*cy
#
#         if abs(z) > 4
#           subspace[i,j] = 1 - exp(-k/10)
#           break
#         end
#
#       end
#
#     end
#   end
#
#   return subspace
# end
#
# """
#   calc_pure_julia() will generate a fractal of the size of a background screen
#   inspecting the complex space bounded by the manifold passing through the
#   points z1 = -1.8 -i1.8 and z2 = 1.8 + i1.8
#   Using the common Julia (Mandelbrot) update rule will chek which points will be
#   bounded by a costume value, here we set 2, using a maximu number of iterations
#   set in NUM_MAXITER
#   After getting the super atomic tensor it will be converted to a proper
#   RGB data strutucture using colorview() provided by ImageCore pkg
#
# """
# function calc_pure_julia()
#   ###Create a list of complex coordinates (zs) and complex parameters (cs)
#   ###Build a Julia set and display
#
#   #Width and height are initially set for the screen size
#   w,h = 1920,1080
#
#   #Theta is the angle at which the vector will turn in the complex space
#   #Will be updated randomly
#   theta = 0.0
#
#
#   #Will need after to properly split the screen pixels
#   rows = rand(1:4)
#   cols = rows + div(rows,2)
#
#   #We want to avoid an eccessive number of computations so we initially
#   #Split the screen in more areas than defined
#   gw= div(w,cols)
#   gh = div(h,rows)
#
#   #A TESNOR or more properly an array of matrices
#   #that will represent the pixels plus the threee
#   #RGB channels
#
#   A = zeros(3,h,w)
#
#   #We iterate through the number of columns and rows that'll we updated
#   #at each iteration of the algorithm
#   #See how the (1:gw) notation in julia defines an array
#   #so we are basicallyiterating through vectors in 3D, where we have (X,Y,theta)
#
#   for ki = 1:rows
#     for kj = 1:cols
#
#       theta = rand() * (2*pi)
#
#       I = (ki-1) * gh + (1:gh) #Notice that those two are arrays,in the next
#       J = (kj - 1) * gw + (1:gw) #cycle we will update whole areas of C-plane
#
#       #For each channel of the RGB color map
#       for i = 1:3
#         #The ray that will span the C-vector space
#         r = 0.62 + (1.8 -0.62) * rand()
#
#         #The center of teh sphere that will hit the space
#         cx = r * cos(theta)
#         cy = r * sin(theta)
#
#         #Update theta for the iterations, will move on a 2pi modulus ring
#         #on the space of the RGB channels
#         theta += 2*pi/3 * rand()
#
#         #Finally we update our favorite tensor selceting a precise channel
#         #of a 2D subspace
#         A[i,I, J] = calculate_z_serial(cx, cy, rows = gh, cols = gw, maxiter = NUM_MAXITER)
#
#       end
#     end
#   end
#
#   #Will open a PREEXISTING image called juliaset.png in your home folder
#   #If it doesn't exists just run touch ~/juliaset.png on terminal
#   file = joinpath(ENV["HOME"], "juliaset.png")
#   save(file, colorview(RGB,A))
#
#   #display the image using a minimal image-view
#   #ENV["DISPLAY"] = ":0"
#   #run('feh --bg-max --no-fehbg $file ')
#
#   # println("Length of x : ", "$length(x)")
#   # println("Total elements:", "$length(zs)")
#   # if draw_output
#   #   show_greyscale(output, width,height,maxiter)
#   #   show_false_greyscale(output, width, height,maxiter)
#   # end
#
#   #println("PORCODDIO")
#
# end
#
# start_time = time_ns()
# calc_pure_julia()
# end_time = time_ns()
# secs = (end_time - start_time) / 10^9
# println("Computations and function call took $secs seconds")
# println("I mean python took 3 aeons circa, is Julia much faster?")
