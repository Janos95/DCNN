require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'
require 'almostIdentity'
local nninit= require 'nninit'

-- net = nn.Sequential()
-- net:add(nn.almostIdentity())
-- net:add(nn.DeformableConvolution(3, 6, 5, 5):init('weight',nninit.normal,0,0.01):init('bias', nninit.constant,0.1))  -- 3 input image channels, 6 output channels, 5x5 convolution kernel
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
-- net:add(nn.DeformableConvolution(6, 16, 5, 5):init('weight',nninit.normal,0,0.01):init('bias', nninit.constant,0.1))
-- 
-- net:add(nn.ReLU()) -- non-linearity 
-- net:add(nn.SpatialMaxPooling(2,2,2,2))
-- net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
-- net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.Linear(120, 84))
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.Linear(84, 10))                   
-- net:add(nn.LogSoftMax())     -- converts the output to a log-probability. 
-- 
--         
-- print('Lenet5\n' .. net:__tostring());

net = torch.load('checkpoint3.t7')


trainset = torch.load('cifar10-train-normalized.t7')
testset = torch.load('cifar10-test-normalized.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}


setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

-- input = torch.rand(3,32,32)
-- gradOutput = torch.rand(10)
-- 
-- x = torch.Timer()
-- for i = 1,1 do
--     output = net:forward(input)
--     net:backward(input,gradOutput)
-- end
-- netElapsedTime = x:time().real
-- --netpara, netgradpara = net:getParameters()
-- netgradinput = net:updateGradInput(input,gradOutput)
-- 
-- --print(netgradinput)

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 55 
trainer.hookIteration = function()
    print('Saving the model...')
    net:clearState()
    torch.save('checkpoint4.t7', net)
    print('Done.')
end
print(trainset)
--trainer:train(trainset)

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true) 
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

--checkpoint2: test accuracy after 11 epochs:    5058    50.58 %
--checkpoint4: test accuracy after 29 epochs:    5168    51.68 % 


print(correct, 100*correct/10000 .. ' % ')






