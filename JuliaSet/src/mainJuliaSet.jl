using Images
using Colors

NUM_MAXITER = 5

##TODO timefn() takes another function as argument and returns
##the time intercurred between the call

#function timefn(fun)
#  t1 = time()
#  result = fun(*args, **kwargs)
#  t2 = time()
#  println
#end

# function show_greyscale(output_raw, width, height, maxiter)
#   ##Convert list to array, show using ++IMAGE-LIBRARY++
#   #scal of gray should be [0...255]
#
#   maxiter = float(max(output_raw))
#   println(maxiter)
#   scale_factor = float(maxiter)
#
#   scaled = [int( o / scale_factor * 255) for o in output_raw]
#
#   #TODO transform scaled into an unsigned array
#   #or anything else that can be displayed with ++IMAGE-LIBRARY++
#
#   #TODO display with ++IMAGE-LIBRARY++
#

# function show_false_greyscale(output_raw, width, height, maxiter)
#   ##Convert list to array, show using ++IMAGE-LIBRARY++
#   #convert input to IMAGE-LIBRARY compatible input
#
#   #sanity check our 1D array and desired 2D form
#   @assert width * height == length(output_raw)
#
#   #rescale output_raw to be in inclusive range [0...255]
#   max_value = float(max(output_raw))
#   output_raw_limited = [(int(float(o) / max_value * 255) for o in output_raw)]
#
#   #NOTTODO #create a slightly fancy colour map that shows colour changes with
#   #increased contrast
#   output_rgb =((o + (256 * o) + (256 ** 2) * o) * 16 for o in output_raw_limited)
#   #thanks to somebody on github <3
#
#   #array of unsigned ints
#   output_rgb = Array{UInt64}(output_rgb)
#
#   #TODO#display with IMAGE-LIBRARY
#   #TODO create a matrix-like image (width,height)
#   #TODO using like PIL.frombytes()
#   #TODO show image
# end

"""
  The function will get as an input the center of the area to be inspected
  the size of the subspace to be inspacted and the maximum number of
  iterations before the matrix value is updated, that is before choosing
  if the series diverges or not. The value of run out (ie if superated)
  we can assume that it tends to infinity is 4
  The output is amatrix of values that corresponds to the values of the
  selected RGB channel that range between [0-1] the first that the runaway
  point is reached the brighter will be that channel (inv proportionality)

"""
function calculate_z_serial(cx, cy; rows = 100, cols = 100, maxiter =10)
###Calcule output matrix using the Julia update rule

  #Center of tha rea in x, ray of the space in y
  Cx = 0.0
  l = 2.0
  r = l * (cols/rows)

  y = linspace(-l, l, rows)
  x = linspace(Cx- r, Cx + r, cols)

  #This subspace will be updated and returned at the end of the loop
  subspace = zeros(rows,cols)

  #Begin the loop on the x-axis
  for i = 1:rows

    #The vector origin to be inspected
    y0 = y[rows - i + 1]

    #Iterate through the y points to check them singularly
    for j = 1:cols
      x0 = x[j]
      z = x0 + im*y0

      for k = 0:maxiter
        z = z^2 + cx + im*cy

        if abs(z) > 4
          subspace[i,j] = 1 - exp(-k/10)
          break
        end

      end

    end
  end

  return subspace
end

"""
  calc_pure_julia() will generate a fractal of the size of a background screen
  inspecting the complex space bounded by the manifold passing through the
  points z1 = -1.8 -i1.8 and z2 = 1.8 + i1.8
  Using the common Julia (Mandelbrot) update rule will chek which points will be
  bounded by a costume value, here we set 2, using a maximu number of iterations
  set in NUM_MAXITER
  After getting the super atomic tensor it will be converted to a proper
  RGB data strutucture using colorview() provided by ImageCore pkg

"""
function calc_pure_julia()
  ###Create a list of complex coordinates (zs) and complex parameters (cs)
  ###Build a Julia set and display

  #Width and height are initially set for the screen size
  w,h = 1920,1080

  #Theta is the angle at which the vector will turn in the complex space
  #Will be updated randomly
  theta = 0.0


  #Will need after to properly split the screen pixels
  rows = rand(1:4)
  cols = rows + div(rows,2)

  #We want to avoid an eccessive number of computations so we initially
  #Split the screen in more areas than defined
  gw= div(w,cols)
  gh = div(h,rows)

  #A TESNOR or more properly an array of matrices
  #that will represent the pixels plus the threee
  #RGB channels

  A = zeros(3,h,w)

  #We iterate through the number of columns and rows that'll we updated
  #at each iteration of the algorithm
  #See how the (1:gw) notation in julia defines an array
  #so we are basicallyiterating through vectors in 3D, where we have (X,Y,theta)

  for ki = 1:rows
    for kj = 1:cols

      theta = rand() * (2*pi)

      I = (ki-1) * gh + (1:gh) #Notice that those two are arrays,in the next
      J = (kj - 1) * gw + (1:gw) #cycle we will update whole areas of C-plane

      #For each channel of the RGB color map
      for i = 1:3
        #The ray that will span the C-vector space
        r = 0.62 + (1.8 -0.62) * rand()

        #The center of teh sphere that will hit the space
        cx = r * cos(theta)
        cy = r * sin(theta)

        #Update theta for the iterations, will move on a 2pi modulus ring
        #on the space of the RGB channels
        theta += 2*pi/3 * rand()

        #Finally we update our favorite tensor selceting a precise channel
        #of a 2D subspace
        A[i,I, J] = calculate_z_serial(cx, cy, rows = gh, cols = gw, maxiter = NUM_MAXITER)

      end
    end
  end

  #Will open a PREEXISTING image called juliaset.png in your home folder
  #If it doesn't exists just run touch ~/juliaset.png on terminal
  file = joinpath(ENV["HOME"], "juliaset.png")
  save(file,colorview(RGB,A))

  #display the image using a minimal image-view
  #ENV["DISPLAY"] = ":0"
  #run('feh --bg-max --no-fehbg $file ')

  # println("Length of x : ", "$length(x)")
  # println("Total elements:", "$length(zs)")
  # if draw_output
  #   show_greyscale(output, width,height,maxiter)
  #   show_false_greyscale(output, width, height,maxiter)
  # end

  #println("PORCODDIO")

end

start_time = time_ns()
calc_pure_julia()
end_time = time_ns()
secs = (end_time - start_time) / 10^9
println("Computations and function call took $secs seconds")
println("I mean python took 3 aeons circa, is Julia much faster?")
