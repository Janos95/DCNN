


function ker2col(kernel, kW, kH)
    output = torch.Tensor(kernel:size(1), kernel:size(2)*kernel:size(3)*kernel:size(4))
    for c2 = 1, kernel:size(1) do
        for c1 = 0, kernel:size(2)-1 do
            for k = 0, kW - 1 do
                for l = 1, kH do
                    output[c2][c1*kW*kH+k*kH+l] = kernel[c2][c1+1][l][k+1]
                end
            end
        end
    end
    return output
end


image = torch.rand(2,3,2,2)
print(image)
print(ker2col(image,2,2))



