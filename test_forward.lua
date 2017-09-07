require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'
local nninit= require 'nninit'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,3,3):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5 

net_new = nn.Sequential()
net_new:add(nn.DeformableConvolution(3,6,3,3):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5



--trainset = torch.load('cifar10-train-normalized.t7')
--testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
input = torch.rand(3,6,8)

gradOutput = torch.rand(6,4,6)

net:zeroGradParameters()


output_new = net_new:forward(input)
output = net:forward(input)

print(net:updateGradInput(input,gradOutput))
print(net_new:updateGradInput(input,gradOutput))


