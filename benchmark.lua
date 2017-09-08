require 'nn'
require 'paths'
require 'image'
require 'SlowSpatialConvolution'

local nninit= require 'nninit'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,5,5):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5 

net_new = nn.Sequential()
net_new:add(nn.SlowSpatialConvolution(3,6,5,5):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5



--trainset = torch.load('cifar10-train-normalized.t7')
--testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
input = torch.rand(3,32,32)


net:zeroGradParameters()

local x = torch.Timer()
for i = 1,10 do
output_new = net_new:forward(input)
end
netNewElapsedTime = x:time().real

x = torch.Timer()
for i = 1,10 do
output = net:forward(input)
--net:backward(input,gradOutput)
end
netElapsedTime = x:time().real

print(string.format("elapsed time for new net: %.2f\n", netNewElapsedTime))
print(string.format("elapsed time for new net: %.2f\n", netElapsedTime))
