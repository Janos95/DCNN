require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'
require 'almostIdentity'
local nninit= require 'nninit'

--visdom = require 'visdom' 
--vis = visdom{server = 'http://localhost', port = 2222, ipv6=false, env=save_str}

net = torch.load('mnist_checkpoint1.t7')

trainset = torch.load('mnist_trainset.t7') 
trainset_normalized = torch.load('mnist_trainset_normalized.t7') 

y = 5
x = 5

point = torch.Tensor(16,8,8):zero()
point[{{},{y},{x}}] = 1

netSplit = nn.Sequential()
netSplit:add(net:get(2))
netSplit:add(net:get(3))
netSplit:add(net:get(4))
netSplit:add(net:get(5))

rd = torch.random(1,60000)

example = trainset.data[rd]
example_normalized = trainset_normalized.data[rd]

netSplit:forward(example_normalized)
sampling = netSplit:backward(example_normalized,point)
sampling = torch.abs(sampling)
maximum = sampling:max()
sampling:div(maximum)
sampling = sampling * 255

classes = {'1','2','3','4','5','6','7','8','9','10'}
local prediction = net:forward(example_normalized)
local confidences, indices = torch.sort(prediction, true) 
print(classes[indices[1]]-1)

example_rgb = torch.Tensor(3,28,28)
example_rgb[1] = example:clone()
example_rgb[2] = example:clone()
example_rgb[3] = example:clone()



samplingScaled = image.scale(sampling, 150, 150,bilinear)
example_rgb_scaled = image.scale(example_rgb, 150, 150,bilinear)


for i = 1, samplingScaled:size(2) do
    for j = 1, samplingScaled:size(3) do 
        if( samplingScaled[1][i][j] > 50) then
            example_rgb_scaled[1][i][j] = samplingScaled[1][i][j]
            example_rgb_scaled[2][i][j] = 0
            example_rgb_scaled[3][i][j] = 0
        end
    end
end

x_scaled = torch.round((150/10)*x)
y_scaled = torch.round((150/10)*y)

example_rgb_scaled[{{2},{y_scaled-1,y_scaled+1},{x_scaled-1,x_scaled+1}}] = 255
example_rgb_scaled[{{3},{y_scaled-1,y_scaled+1},{x_scaled-1,x_scaled+1}}] = 0
example_rgb_scaled[{{1},{y_scaled-1,y_scaled+1},{x_scaled-1,x_scaled+1}}] = 0

--vis:image{img = example_rgb_scaled}
image.display(example_rgb_scaled)

print(rd)

