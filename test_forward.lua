require 'nn'
require 'paths'
require 'image'
require 'DeformableConv'
local nninit= require 'nninit'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,2,2):init('weight',nninit.constant,1)) -- 3 input image channels, 6 output channels, 5x5 

net_new = nn.Sequential()
net_new:add(nn.DeformableConv(3,6,2,2):init('weight',nninit.constant,1)) -- 3 input image channels, 6 output channels, 5x5



trainset = torch.load('cifar10-train-normalized.t7')
testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
input = torch.rand(3,8,8)
           
output = net:forward(input)
output_new = net_new:forward(input)

print(output)
print(output_new)



