require 'nn'
require 'DeformableConvolution'
require 'SlowSpatialConvolution'

local mytest1 = torch.TestSuite()
local mytest2 = torch.TestSuite()
local mytest3 = torch.TestSuite()
local mytest4 = torch.TestSuite()
local mytest5 = torch.TestSuite()

local mytester = torch.Tester()

local precision = 1e-5
local jac = nn.Jacobian

local from = 3
local ki = 3
local kj = 3
local inj = 8
local ini = 8



function mytest1.UpdateGradInput()
   local to = math.random(1,5)
   local module = nn.DeformableConvolution(from, to, ki, kj)
   local input = torch.Tensor(from, inj, ini):zero()

   local function test1(module)
      local err, diff = nn.Jacobian.testJacobian(module, input)
      if err >= precision then
        --print(diff)
      end
      mytester:assertlt(err, precision, 'error on gradient w.r.t. input ')
   end
                  
   test1(module)

end

function mytest2.AccGradWeight()
   local to = math.random(1,5)
   local module = nn.DeformableConvolution(from, to, ki, kj)
   local input = torch.Tensor(from, inj, ini):zero()

    
    local function test2(module)
      local err, diff = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
      if err >= precision then
            --print(diff)
            print(diff[{{1, from*ki*kj*to},{}}]:max())
      end
      mytester:assertlt(err , precision, 'error on gradient w.r.t. weight ')
    end

   test2(module)

end

function mytest3.AccGradBias()
   local to = math.random(1,5)
   local module = nn.DeformableConvolution(from, to, ki, kj)
   local input = torch.Tensor(from, inj, ini):zero()

    local function test3(module)
      local err, diff = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
      if( err >= precision) then
          print(diff[{{1,to},{}}]:max())
      end
      mytester:assertlt(err , precision, 'error on gradient w.r.t. bias ')
    end
                     
   test3(module)

end

function mytest4.AccGradWeightDirect()
   local to = math.random(1,5)
   local module = nn.DeformableConvolution(from, to, ki, kj)
   local input = torch.Tensor(from, inj, ini):zero()

    local function test4(module)
      local err = jac.testJacobianUpdateParameters(module, input, module.weight)
      mytester:assertlt(err , precision, 'error on gradient w.r.t. weight [direct update] ')
    end
                     
   test4(module)

end

function mytest5.AccGradBiasDirect()
   local to = math.random(1,5)
   local module = nn.DeformableConvolution(from, to, ki, kj)
   local input = torch.Tensor(from, inj, ini):zero()

    local function test5(module)
      local err,diff = jac.testJacobianUpdateParameters(module, input, module.bias)
         mytester:assertlt(err , precision, 'error on gradient w.r.t. bias [direct update] ')
     if( err >= precision) then
          --print(diff)
      end
    end
                      
   test5(module)

end
    
mytester:add(mytest1)
mytester:add(mytest2)
mytester:add(mytest3)
mytester:add(mytest4)
mytester:add(mytest5)
mytester:run()
