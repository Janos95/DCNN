require 'deformableconvolution'
require 'nn'
local nninit= require 'nninit'

local DeformableConvolution, parent = torch.class('nn.DeformableConvolution', 'nn.Module')

function DeformableConvolution:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)


   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
      

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   self.offsetPredictor = nn.SpatialConvolution(nInputPlane,2*kW*kH*nInputPlane,kW,kH)--:init('weight',nninit.constant,0):init('bias', nninit.constant,0) 
   
end

function DeformableConvolution:updateOutput(input)
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    
    self.output = torch.Tensor(self.nOutputPlane, hOutputImage, wOutputImage)
    offset = self.offsetPredictor:forward(input):view(
        self.nInputPlane
        ,self.kH
        ,self.kW
        ,hOutputImage
        ,wOutputImage
        ,2)
    unfoldedInput = deformableconvolution.im2col(input,offset,self.kH,self.kW)
    self.output = torch.mm(
        self.weight:view(self.nOutputPlane,self.nInputPlane*self.kW*self.kH)
        ,unfoldedInput
    ):view(self.nOutputPlane,hOutputImage, wOutputImage)
    
    for c2 = 1, self.nOutputPlane do
        self.output[c2]:add(self.bias[c2])
    end
    
    return self.output 
end
    
function DeformableConvolution:updateGradInput(input,gradOutput)
    self.gradInput = torch.Tensor(input:size()):zero()
    weightRotated = deformableconvolution.rotate(self.weight)
    fgradOutput = deformableconvolution.frame(gradOutput,self.kH-1,self.kW-1)
    for c1star = 1, self.nInputPlane do
        for c2 = 1, self.nOutputPlane do
            self.gradInput[c1star]:add(torch.mm(
                weightRotated[c2][c1star]:view(1,self.kW*self.kH)
                ,deformableconvolution.im2col(
                    fgradOutput[c2]:view(1,fgradOutput:size(2),fgradOutput:size(3))
                    ,self.kH
                    ,self.kW)
            ):view(input:size(2),input:size(3)))
        end    
    end
    return self.gradInput
end
    
function DeformableConvolution:accGradParameters(input,gradOutput, scale)
    scale = scale or 1
    
    local gradBias = torch.Tensor(self.gradBias:size()):zero()
    local gradWeight = torch.Tensor(self.gradWeight:size()):zero()
    
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    
    ones = torch.Tensor(gradOutput:size(2),gradOutput:size(3)):fill(1)
    for i = 1, self.nOutputPlane do
        gradBias[i] = gradBias[i] + gradOutput[i]:dot(ones)
    end
    
    offsets = ((self.offsetPredictor).output):view(
        self.nInputPlane
        ,self.kH
        ,self.kW
        ,hOutputImage
        ,wOutputImage
        ,2)
    
    for c1star = 1, self.nInputPlane do
        for c2star = 1, self.nOutputPlane do
            gradWeight[c2star][c1star]:add(torch.mm(
                gradOutput[c2star]:view(1,gradOutput:size(2)*gradOutput:size(3))
                ,deformableconvolution.im2col(
                    input[c1star]:view(1,input:size(2),input:size(3))
                    ,offsets[c1star]:view(
                        1
                        ,offsets:size(2)
                        ,offsets:size(3)
                        ,offsets:size(4)
                        ,offsets:size(5)
                        ,offsets:size(6))
                    ,gradOutput:size(2)
                    ,gradOutput:size(3))
            ):view(self.kH,self.kW))
        end
    end
    
    self.gradBias:add(scale, gradBias)
    self.gradWeight:add(scale, gradWeight)
end


    
    
    
    
        
                    
            
    
    
        
