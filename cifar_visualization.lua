require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'
require 'almostIdentity'
local nninit= require 'nninit'
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

visdom = require 'visdom' 
vis = visdom{server = 'http://localhost', port = 2222, ipv6=false, env=save_str}

net = torch.load('checkpoint4.t7')

trainset = torch.load('cifar10-train.t7')
trainset_normalized = torch.load('cifar10-train-normalized.t7')

trainset_normalized.data = trainset_normalized.data:double()
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

rd = torch.random(1,10000)
rd = 611

example = trainset.data[rd]
exampleNormalized = trainset_normalized.data[rd]

y = 9
x = 7

point = torch.Tensor(16,10,10):zero()
point[{{},{y},{x}}] = 1

netSplit = nn.Sequential()
netSplit:add(net:get(2))
netSplit:add(net:get(3))
netSplit:add(net:get(4))
netSplit:add(net:get(5))


netSplit:forward(exampleNormalized)
sampling = netSplit:backward(exampleNormalized,point)
sampling = torch.abs(sampling)
sampling = sampling[1] + sampling[2] + sampling[3]
maximum = sampling:max()
sampling:div(maximum)
sampling = sampling * 255


classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
           
local prediction = net:forward(exampleNormalized)
local confidences, indices = torch.sort(prediction, true) 
print(classes[indices[1]])


samplingScaled = image.scale(sampling, 150,150,bilinear)
example_scaled = image.scale(example,150,150,bilinear)

orig = example_scaled:clone()

for i = 1, 150 do
    for j = 1, 150 do 
        if( samplingScaled[i][j] > 50) then
            example_scaled[1][i][j] = samplingScaled[i][j]
            example_scaled[2][i][j] = 0
            example_scaled[3][i][j] = 0
        end
    end
end

y_scaled = y*(150/10)
x_scaled = x*(150/10)



example_scaled[{{2},{y_scaled-1,y_scaled+1},{x_scaled-1,x_scaled+1}}] = 255
example_scaled[{{1},{y_scaled-1,y_scaled+1},{x_scaled-1,x_scaled+1}}] = 0
example_scaled[{{3},{y_scaled-1,y_scaled+1},{x_scaled-1,x_scaled+1}}] = 0

image.display(example_scaled)

print(rd)





--vis:image{img = horseOriginal}
