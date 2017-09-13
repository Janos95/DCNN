require 'deformableconvolution'

input = torch.Tensor(2,2) 
input[1][1] = 1
input[1][2] = 2
input[2][1] = 3
input[2][2] = 4

output = deformableconvolution.bilinearInterpolation(input:view(1,2,2),1,1,2)

print(output)
