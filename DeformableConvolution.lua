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
   self.bufferIndices = torch.LongTensor()
   self.bufferInterpolationWeights = torch.Tensor()
   
   self.offsetPredictor = nn.SpatialConvolution(nInputPlane,2*kW*kH*nInputPlane,kW,kH):init('weight',nninit.constant,0):init('bias', nninit.constant,0)
   
end

function DeformableConvolution:updateOutput(input)
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    
    self.bufferIndices:resize(self.kW*self.kH*self.nInputPlane, wOutputImage*hOutputImage)
    self.bufferInterpolationWeights:resize(4,self.kW*self.kH*self.nInputPlane, wOutputImage*hOutputImage)
    self.output:resize(self.nOutputPlane, hOutputImage, wOutputImage)
    
    offset = self.offsetPredictor:forward(input):view(
        self.nInputPlane
        ,hOutputImage
        ,wOutputImage
        ,self.kH
        ,self.kW
        ,2)
    unfoldedInput = deformableconvolution.im2col(input,offset,self.kH,self.kW,self.bufferIndices,self.bufferInterpolationWeights,1)
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
    local gradInput = torch.Tensor(input:size()):zero()
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    local gradIm2col = torch.Tensor(self.kW*self.kH*input:size(1),wOutputImage*hOutputImage)
    gradIm2col = torch.mm(
        self.weight:view(
            self.nOutputPlane
            ,self.nInputPlane*self.kW*self.kH):transpose(1,2)
        ,gradOutput:view(self.nOutputPlane, gradOutput:size(2)*gradOutput:size(3)))
    self.gradInput = deformableconvolution.update_grad_input(gradInput,gradIm2col,self.bufferIndices,self.bufferInterpolationWeights) 

    return self.gradInput
end
    
function DeformableConvolution:accGradParameters(input, gradOutput, scale)
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
        ,hOutputImage
        ,wOutputImage
        ,self.kH
        ,self.kW
        ,2)
    for c1star = 1, self.nInputPlane do
        for c2star = 1, self.nOutputPlane do
            --[[print(hOutputImage, wOutputImage, self.kH, self.kW)
            print(deformableconvolution.im2col(
                    input[c1star]:view(1,input:size(2),input:size(3))
                    ,offsets[c1star]:view(
                        1
                        ,offsets:size(2)
                        ,offsets:size(3)
                        ,offsets:size(4)
                        ,offsets:size(5)
                        ,offsets:size(6))
                    ,gradOutput:size(2)
                    ,gradOutput:size(3)
                    ,torch.LongTensor() -- empty long tensor for buffer indices
                    ,torch.Tensor() -- empty double tensor for buffer
                    ,0):size())
            print(gradOutput[c2star]:view(1,gradOutput:size(2)*gradOutput:size(3)):size())]]        
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
                    ,gradOutput:size(3)
                    ,torch.LongTensor() -- empty long tensor for buffer indices
                    ,torch.Tensor() -- empty double tensor for buffer
                    ,0)
            ):view(self.kH,self.kW))
        end
            
    end
    
    self.gradBias:add(scale, gradBias)
    self.gradWeight:add(scale, gradWeight)
end


    
    
    
    
        
                    
            
    
    
        
