
function updateOutput(input)
    nOutputPlane = 6
    nInputPlane = 4
    output = torch.Tensor(nOutputPlane, input:size(2),input:size(3))
    for i=1,nOutputPlane do
        for j=1,nInputPlane do
            output[i] = output[i]+convolution(weight[i][j],input[j],3,3)
        end
        output[i] = output[i]
    end
    return output
end

function convolution(kernel, image, kW,kH)
    output_image = torch.Tensor(image:size()):zero()
    local w = image:size(1)
    local h = image:size(2)
    local kW = kW
    local kH = kH
    local padW = math.floor(kW/2)
    local padH = math.floor(kH/2)
    image_framed = torch.Tensor(w+2*padW,h+2*padH):zero()
    for i = 1,w do
        for j=1,h do
            image_framed[i+padW][j+padH] = image[i][j]
        end
    end
    
    for i=1,w do
        for j=1,h do
            for k=0,kW-1 do
                for l=0,kH-1 do
                    output_image[i][j] = output_image[i][j] + image_framed[i+k][j+l]*kernel[k+1][l+1]
                end
            end
        end
    end
    return output_image
end

weight = torch.Tensor(6,4,3,3):fill(2)

image = torch.Tensor(4,5,5):fill(1)

print(updateOutput(image))


