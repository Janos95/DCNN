require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'
local nninit= require 'nninit'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,5,5,nil,nil,1):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5 

net_new = nn.Sequential()
net_new:add(nn.DeformableConvolution(3,6,5,5,1):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5



--trainset = torch.load('cifar10-train-normalized.t7')
--testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
input = torch.rand(3,8,8)

function rotate(input)
    output = torch.Tensor(input:size())
    for i = 1, input:size(1) do
        for j = 1, input:size(2) do
            for k = 1, input:size(3) do 
                output[i][j][k] = input[i][input:size(2)-i+1][input:size(3)-j+1]
            end
        end
    end
    return output
end
        

print(input)
print(rotate(input))
           




