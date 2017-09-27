local mnist = require 'mnist'

local trainingset = mnist.traindataset()
local testingset = mnist.testdataset()



datah = torch.DoubleTensor(60000,1,28,28)
labell = torch.ByteTensor(60000)
for i = 1, 60000 do
    datah[i] = trainingset[i].x:double():view(1,28,28):clone()
    labell[i] = trainingset[i].y
end

datahh = torch.DoubleTensor(10000,1,28,28)
labelll = torch.ByteTensor(10000)
for i = 1, 10000 do
    datahh[i] = testingset[i].x:double():view(1,28,28):clone()
    labelll[i] = testingset[i].y
end    

trainset = {
    size = 60000,
    data = datah,
    label = labell +1 
}

testset = {
    size = 10000,
    data = datahh,
    label = labelll+1
}

torch.save('mnist_trainset.t7',trainset)
torch.save('mnist_testset.t7',testset)

mean = 0 -- store the mean, to normalize the test set in the future
stdv  = 0 -- store the standard-deviation for the future
mean = trainset.data:mean() -- mean estimation
trainset.data:add(-mean) -- mean subtraction
testset.data:add(-mean)

stdv = trainset.data:std() -- std estimation
trainset.data:div(stdv) -- std scaling
testset.data:div(stdv)

torch.save('mnist_trainset_normalized.t7',trainset)
torch.save('mnist_testset_normalized.t7',testset)
