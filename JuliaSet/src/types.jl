using Colors
using FixedPointNumbers

#An RGB Type made of 8-bit flpats that for each channel
#represent the intensity [0..1]

typealias RGB8bit ColorTypes.RGB{FixedPointNumbers.UFixed{UInt8,8}}

#Considering the image as a function, the gradient represents the
#derivative(jacobian) or the inst change

type Gradient
    startcolor::RGB8bit
    endcolor::RGB8bit
    nicolor::RGB8bit
end

#

type ImageSpec
    width::Int64
    height::Int64
    num_supersamples::Int64
end

#Primarly defines the region of the complex plane to investigate

type CameraSpec
    gradient::Gradient
    center::Complex
    zoom::Float64
    pow::Float64
    affine::bool
end
