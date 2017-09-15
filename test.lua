require 'nn'
require 'DeformableConvolution'
require 'SlowSpatialConvolution'

local mytest = torch.TestSuite()

local mytester = torch.Tester()

local precision = 1e-5
local jac = nn.Jacobian



function mytest.DeformableConvolution()
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   local kj = math.random(1,5)
   local outi = math.random(5,7)
   local outj = math.random(5,7)
   local ini = (outi-1)+ki
   local inj = (outj-1)+kj
   local module = nn.DeformableConvolution(from, to, ki, kj)
   local input = torch.Tensor(from, inj, ini):zero()

   local function jacTests(module)
      -- stochastic

--       local err = nn.Jacobian.testJacobian(module, input)
--       mytester:assertlt(err, precision, 'error on gradient w.r.t. input ')

      local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
      mytester:assertlt(err , precision, 'error on gradient w.r.t. weight ')

      if module.bias then
         local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
         mytester:assertlt(err , precision, 'error on gradient w.r.t. bias ')
      end

      local err = jac.testJacobianUpdateParameters(module, input, module.weight)
      mytester:assertlt(err , precision, 'error on gradient w.r.t. weight [direct update] ')

      if module.bias then
         local err = jac.testJacobianUpdateParameters(module, input, module.bias)
         mytester:assertlt(err , precision, 'error on gradient w.r.t. bias [direct update] ')
      end

    end
                      
   jacTests(module)

end
    
mytester:add(mytest)
mytester:run()
