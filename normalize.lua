require 'paths'

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
testset.data = testset.data:double()

function trainset:size() 
    return self.data:size(1) 
end



mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

torch.save('cifar10-train-normalized.t7',trainset)
torch.save('cifar10-test-normalized.t7',testset)
