require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'

local nninit= require 'nninit'

w = 32
h = 32

nInputPlane = 3
nOutputPlane = 6
kW = 5
kH = 5
scale = 0.001

weights = torch.rand(nOutputPlane*nInputPlane*kH*kW+nInputPlane*2*kH*kW*kH*kW)
bias = torch.rand(nOutputPlane+2*kH*kW)

-- net = nn.Sequential()
-- net:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,kW,kH):init('weight',nninit.copy,weights):init('bias', nninit.copy,bias)) 

net_new = nn.Sequential()
net_new:add(nn.DeformableConvolution(nInputPlane,nOutputPlane,kW,kH):init('weight',nninit.copy,weights):init('bias', nninit.copy,bias)) 



--trainset = torch.load('cifar10-train-normalized.t7')
--testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
input = torch.Tensor(nInputPlane,h,w):fill(1)
gradOutput =torch.rand(nOutputPlane,h-kH+1,w-kW+1)

local x = torch.Timer()
for i = 1,1 do
output_new=net_new:forward(input)
net_new:backward(input, gradOutput)
end
netNewElapsedTime = x:time().real
netnewpara, netnewgradpara = net_new:getParameters()
netnewgradinput = net_new:updateGradInput(input,gradOutput)

-- x = torch.Timer()
-- for i = 1,1 do
-- output = net:forward(input)
-- net:backward(input,gradOutput)
-- end
-- netElapsedTime = x:time().real
-- netpara, netgradpara = net:getParameters()
-- netgradinput = net:updateGradInput(input,gradOutput)

-- print(output - output_new)



print(string.format("elapsed time for new net: %.2f\n", netNewElapsedTime))
--print(string.format("elapsed time for net: %.2f\n", netElapsedTime))

