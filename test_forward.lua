require 'nn'
require 'paths'
require 'image'
require 'SlowSpatialConvolution'
local nninit= require 'nninit'
require 'slowspatialconvolution'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3,6,4,4):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5 

net_new = nn.Sequential()
net_new:add(nn.SlowSpatialConvolution(3,6,4,4):init('weight',nninit.constant,1):init('bias', nninit.constant,0)) -- 3 input image channels, 6 output channels, 5x5



--trainset = torch.load('cifar10-train-normalized.t7')
--testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
input = torch.rand(1,1,4,5)





