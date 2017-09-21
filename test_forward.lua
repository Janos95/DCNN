require 'nn'
require 'paths'
require 'image'
require 'deformableconvolution'
local nninit= require 'nninit'



           
input = torch.DoubleTensor(1,2,2)

input[1][1][1] = 1
input[1][1][2] = 2
input[1][2][1] = 3
input[1][2][2] = 4


epsilon = 10e-5

del_x = 
(deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,2.5,2.5+epsilon,0,0,0)-
deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,2.5,2.5-epsilon,0,0,0))/(2*epsilon
)

del_y = 
(deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,2.5+epsilon,2.5,0,0,0)-
deformableconvolution.bilinearInterpolation(input,torch.LongTensor(),torch.Tensor(),1,2.5-epsilon,2.5,0,0,0))/(2*
epsilon )

w0 = 0
w1 = 0
w2 = 1
w3 = 1

vy = -input[1][1][1]*w1 - input[1][1][2]*w3
 +input[1][2][1]*w1
+input[1][2][2]*w3

vx = -input[1][1][1]*w0 + input[1][1][2]*w0
 -input[1][2][1]*w2
+input[1][2][2]*w2


print(del_x, vx)
print(del_y,vy)




