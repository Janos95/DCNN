require 'nn'
require 'paths'
require 'image'
require 'DeformableConvolution'
require 'almostIdentity'
local nninit= require 'nninit'

classes = {'1','2','3','4','5','6','7','8','9','10'}

model = nn.Sequential()

-- net = nn.Sequential()
-- net:add(nn.almostIdentity())
-- net:add(nn.SpatialConvolution(1, 6, 5, 5):init('weight',nninit.normal,0,0.01):init('bias', nninit.constant,0))  -- 3 input image channels, 6 output channels, 5x5 
-- --convolution kernel
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
-- net:add(nn.SpatialConvolution(6, 16, 5, 5):init('weight',nninit.normal,0,0.01):init('bias', nninit.constant,0))
-- 
-- net:add(nn.ReLU()) -- non-linearity 
-- net:add(nn.SpatialMaxPooling(2,2,2,2))
-- net:add(nn.View(16*4*4))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
-- net:add(nn.Linear(16*4*4, 120))             -- fully connected layer (matrix multiplication between input and weights)
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.Linear(120, 84))
-- net:add(nn.ReLU())                       -- non-linearity 
-- net:add(nn.Linear(84, 10))                   
-- net:add(nn.LogSoftMax())     -- converts the output to a log-probability. 
-- 
--         
-- print('Lenet5\n' .. net:__tostring());

net = torch.load('mnist_checkpoint1.t7')

trainset = torch.load('mnist_trainset_normalized.t7')
testset = torch.load('mnist_testset_normalized.t7')

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

function trainset:size() 
    return self.data:size(1) 
end

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 100
trainer.hookIteration = function()
    print('Saving the model...')
    net:clearState()
    torch.save('mnist_checkpoint2.t7', net)
    print('Done.')
end

correct = 0
for i=1,10000 do
    print(i)
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i]) 
    local confidences, indices = torch.sort(prediction, true) 
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

trainer:train(trainset)

correct = 0
for i=1,10000 do
    print(i)
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i]) 
    local confidences, indices = torch.sort(prediction, true) 
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')

-- # current error = 0.057080773293819
-- # precision after 3 epochs = 98.35%






