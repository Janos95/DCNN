function rotate(input)
    output = torch.Tensor(input:size())
    for i = 1, input:size(1) do
        for j = 1, input:size(2) do
            for k = 1, input:size(3) do 
                output[i][j][k] = input[i][input:size(2)-j+1][input:size(3)-k+1]
            end
        end
    end
    return output
end

a = torch.rand(1,2,3)
b = rotate(a)
print(a)
print(b)
