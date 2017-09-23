require 'nn'
require 'paths'
require 'image'
require 'almostIdentity'
require 'deformableconvolution'
require 'DeformableConvolution'
local nninit= require 'nninit'


--net = nn.Sequential()
net = torch.load('checkpoint.t7')

testset = torch.load('cifar10-test-normalized.t7')
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor

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






























--input = torch.DoubleTensor(1,2,2)

-- input[1][1][1] = 1
-- input[1][1][2] = 2
-- input[1][2][1] = 3
-- input[1][2][2] = 4

-- epsilon = 10e-5

-- del_x = 
-- (deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,2.5,2.5+epsilon,0,0,0)-
-- deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,2.5,2.5-epsilon,0,0,0))/(2*epsilon
-- )



-- for i = 1,100 do
-- x = torch.random(0,300)/100    
-- y = torch.random(0,300)/100
-- 
-- del_y = 
-- (deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,y+epsilon,x,0,0,0)-
-- deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,y-epsilon,x,0,0,0))/(2*
-- epsilon)
-- 
-- del_x = 
-- (deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,y,x+epsilon,0,0,0)-
-- deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,y,x-epsilon,0,0,0))/(2*epsilon
-- )
-- 
-- x_proj = x
-- y_proj = y
-- 
-- if( x > 2) then
--     x_proj = 2
-- end
-- if( x < 1) then
--     x_proj=1
-- end
-- if( y > 2) then
--     y_proj = 2
-- end
-- if( y < 1) then
--     y_proj=1
-- end
-- 
-- w0 = 1 - torch.abs(1 -y_proj) 
-- w1 = 1 - torch.abs(x_proj-1)
-- w2 = 1 - torch.abs(2-y_proj)
-- w3 = 1 - torch.abs(2-x_proj)
-- 
-- vy=0
-- vx=0
-- 
-- if( x == x_proj or y == y_proj) then
--     vy = -input[1][1][1]*w1 - input[1][1][2]*w3
--  +input[1][2][1]*w1
-- +input[1][2][2]*w3
-- vx = -input[1][1][1]*w0 + input[1][1][2]*w0
--  -input[1][2][1]*w2
-- +input[1][2][2]*w2
-- end
-- 
--     if(x ~= x_pro) then
--         
--     
--     
-- if(x ~= x_proj) then
-- vy = -input[1][1][1]*w1 - input[1][1][2]*w3
--  +input[1][2][1]*w1
-- +input[1][2][2]*w3
-- end
-- 
-- if(y ~= y_proj) then
-- vx = -input[1][1][1]*w0 + input[1][1][2]*w0
--  -input[1][2][1]*w2
-- +input[1][2][2]*w2
-- end
-- 
-- if( x ~= x_proj and y ~= y_proj) then
--     print("projected twice")
--     vy = 0
--     vx = 0
-- end
-- 
-- if( x == x_proj and y == y_proj) then
-- vy = -input[1][1][1]*w1 - input[1][1][2]*w3
--  +input[1][2][1]*w1
-- +input[1][2][2]*w3
-- vx = -input[1][1][1]*w0 + input[1][1][2]*w0
--  -input[1][2][1]*w2
-- +input[1][2][2]*w2
-- end
-- 
-- err_x = torch.abs(del_x - vx)
-- err_y = torch.abs(del_y - vy)
-- if(err_x > 0.1 or err_y > 0.1) then
--     print("point", y,x)
--     print("error", err_y, err_x)
--     print("symbolic" , vy,vx)
--     print("numeric" , del_y,del_x)
--     if( vx == 0 and vy == 0) then
--         print(w0,w1,w2,w3)
--     end
-- end
-- print(i)
-- end



