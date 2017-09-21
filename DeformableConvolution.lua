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
     
   self.randomTensor = torch.rand(2,3,3,6,6)
   
   self.weight = torch.randn(nOutputPlane*nInputPlane*kH*kW+nInputPlane*2*kH*kW*kH*kW)
   self.bias = torch.Tensor(nOutputPlane+2*kH*kW):zero()
   
   self.gradWeight = torch.Tensor(nOutputPlane*nInputPlane*kH*kW+nInputPlane*2*kH*kW*kH*kW):zero()
   self.gradBias = torch.Tensor(nOutputPlane+2*kH*kW):zero()
   
   self.weightDC = torch.Tensor(self.weight:storage(),1,torch.LongStorage{nOutputPlane,nInputPlane,kH,kW})
   self.biasDC = torch.Tensor(self.bias:storage(),1,torch.LongStorage{nOutputPlane})
   
   self.gradWeightDC = torch.Tensor(self.gradWeight:storage(),1,torch.LongStorage{nOutputPlane,nInputPlane,kH,kW})
   self.gradBiasDC = torch.Tensor(self.gradBias:storage(),1,torch.LongStorage{nOutputPlane})
   
   self.bufferIndices = torch.LongTensor()
   self.bufferInterpolationWeights = torch.Tensor()
   
   self.offsetPredictor = 
nn.SpatialConvolution(nInputPlane,2*kH*kW,kW,kH)

   self.offsetPredictor.weight= 
torch.Tensor(self.weight:storage(),1+nOutputPlane*nInputPlane*kH*kW,torch.LongStorage{2*kH*kW,nInputPlane,kH,kW})
   self.offsetPredictor.bias 
=torch.Tensor(self.bias:storage(),1+nOutputPlane,torch.LongStorage{2*kH*kW})

   self.offsetPredictor.gradWeight = 
torch.Tensor(self.gradWeight:storage(),1+nOutputPlane*nInputPlane*kH*kW,torch.LongStorage{2*kH*kW,nInputPlane,kH,kW})
   self.offsetPredictor.gradBias = 
torch.Tensor(self.gradBias:storage(),1+nOutputPlane,torch.LongStorage{2*kH*kW})


end

function DeformableConvolution:updateOutput(input)
    
    --print('updateOutput')
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    
    self.bufferIndices:resize(self.nInputPlane*self.kW*self.kH, 
hOutputImage*wOutputImage,3)

    -- Prefill the array so that any mistakes are easier to find
    self.bufferIndices:fill(2000)
    self.bufferInterpolationWeights:resize(self.nInputPlane*self.kW*self.kH, wOutputImage*hOutputImage, 4)
    
    self.output:resize(self.nOutputPlane, hOutputImage, wOutputImage)

    offsets = self.offsetPredictor:forward(input):view(
        2
        ,self.kH
        ,self.kW
        ,hOutputImage
        ,wOutputImage)
        

    
    

    assert(input:isContiguous())
    --unfoldedInput = torch.rand(self.nInputPlane*self.kH*self.kW,hOutputImage*wOutputImage)
    unfoldedInput = 
deformableconvolution.im2col(input,offsets,self.kH,self.kW,self.bufferIndices,self.bufferInterpolationWeights,1)
    self.output = torch.mm(
        self.weightDC:view(self.nOutputPlane,self.nInputPlane*self.kW*self.kH)
        ,unfoldedInput
    ):view(self.nOutputPlane,hOutputImage, wOutputImage)
    
    for c2 = 1, self.nOutputPlane do
        self.output[c2]:add(self.biasDC[c2])
    end
    return self.output 
end
    
function DeformableConvolution:updateGradInput(input,gradOutput)
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1

    gradIm2col = torch.mm(
        self.weightDC:view(
            self.nOutputPlane
            ,self.nInputPlane*self.kW*self.kH):transpose(1,2):contiguous()
        ,gradOutput:view(self.nOutputPlane, 
gradOutput:size(2)*gradOutput:size(3)))
--     print(self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
--     print(gradIm2col:size())
    gradOffset = deformableconvolution.grad_offset(
                            input,
                            offsets,
                            self.weightDC,
                            gradOutput,
                            self.bufferIndices,
                            self.bufferInterpolationWeights)
    self.gradInput = 
deformableconvolution.update_grad_input(gradIm2col,self.bufferIndices, 
self.bufferInterpolationWeights, input:size(1), input:size(2), 
input:size(3)):add(self.offsetPredictor:updateGradInput(input,gradOffset))

    return self.gradInput
end
    
function DeformableConvolution:accGradParameters(input, gradOutput, scale)
    --print('accGradParameters')
    scale = scale or 1
    
    local gradBiasDC = torch.Tensor(self.gradBiasDC:size()):zero()
    local gradWeightDC = torch.Tensor(self.gradWeightDC:size()):zero()
    
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    
    ones = torch.Tensor(gradOutput:size(2),gradOutput:size(3)):fill(1)
    for i = 1, self.nOutputPlane do
        gradBiasDC[i] = gradBiasDC[i] + gradOutput[i]:dot(ones)
    end
    
    offsets = ((self.offsetPredictor).output):view(
        2
        ,self.kH
        ,self.kW
        ,hOutputImage
        ,wOutputImage)
        
    
    
    for c1star = 1, self.nInputPlane do
        for c2star = 1, self.nOutputPlane do
            assert(input[c1star]:view(1,input:size(2),input:size(3)):isContiguous())

            input2col = deformableconvolution.im2col(
                    input[c1star]:view(1,input:size(2),input:size(3))
                    ,offsets:transpose(2,4):transpose(3,5):contiguous()
                    ,gradOutput:size(2)
                    ,gradOutput:size(3)
                    ,torch.LongTensor() -- empty long tensor for buffer indices
                    ,torch.Tensor() -- empty double tensor for buffer
                    ,0) -- dont update the buffer
                    
            assert(input2col:isContiguous())

            gradWeightDC[c2star][c1star]:add(torch.mm(
                gradOutput[c2star]:view(1,gradOutput:size(2)*gradOutput:size(3))
                ,input2col
            ):view(self.kH,self.kW))
        end
            
    end
    
    gradOffset = deformableconvolution.grad_offset(
                            input,
                            offsets,
                            self.weightDC,
                            gradOutput,
                            self.bufferIndices,
                            self.bufferInterpolationWeights)
                                                
    self.offsetPredictor:accGradParameters(input, gradOffset, scale)
                            
    self.gradBiasDC:add(scale, gradBiasDC)
    self.gradWeightDC:add(scale, gradWeightDC)
       
end




    
    
    
    
        
                    
            
    
    
        
