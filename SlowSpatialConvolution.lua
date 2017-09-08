
require 'slowspatialconvolution'

local SlowSpatialConvolution, parent = torch.class('nn.SlowSpatialConvolution', 'nn.Module')

function SlowSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)


   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
      

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

end


function SlowSpatialConvolution:im2col(image)
    local wOutputImage = image:size(3)-self.kW+1
    local hOutputImage = image:size(2)-self.kH+1
    outputImage = torch.Tensor(self.kW*self.kH*image:size(1), wOutputImage*hOutputImage)
    for c = 0, image:size(1)-1 do
        for k = 0, self.kW - 1 do
            for l = 1, self.kH do
                for i = 0, hOutputImage - 1 do
                    for j = 1, wOutputImage do
                      outputImage[c*self.kW*self.kH+k*self.kH+l][i*wOutputImage+j] = image[c+1][i+l][j+k]
                    end
                end
            end
        end
    end
    return outputImage
end

function SlowSpatialConvolution:updateOutput(input)
    local wOutputImage = input:size(3)-self.kW+1
    local hOutputImage = input:size(2)-self.kH+1
    self.output = torch.Tensor(self.nOutputPlane, hOutputImage, wOutputImage)
    unfoldedInput = slowspatialconvolution.im2col(input,self.kW,self.kH)
    self.output = torch.mm(self.weight:view(self.nOutputPlane,self.nInputPlane*self.kW*self.kH),unfoldedInput):view(self.nOutputPlane,hOutputImage, wOutputImage)
    return self.output
end
    
 

function SlowSpatialConvolution:updateGradInput(input,gradOutput)
    self.gradInput = torch.Tensor(input:size()):zero()
    for c1star = 1,self.nInputPlane do
        for istar = 1,input:size(2) do
            for jstar = 1,input:size(3) do
                for i = 1,gradOutput:size(2) do
                    for j = 1,gradOutput:size(3) do 
                        for c2 =1,self.nOutputPlane do
                            k = istar + 1 - i
                            l = jstar + 1 - j
                            if(1 <= l and l <= self.kW and 1 <= k and k <= self.kH) then
                                self.gradInput[c1star][istar][jstar] = self.gradInput[c1star][istar][jstar] + gradOutput[c2][i][j]*self.weight[c2][c1star][k][l]
                            end
                        end
                    end
                end
            end
        end
    end
    return self.gradInput
end
    
    
function SlowSpatialConvolution:accGradParameters(input,gradOutput, scale)
    self.gradWeight:zero()
    self.gradBias:zero()
    ones = torch.Tensor(gradOutput:size(2),gradOutput:size(3)):fill(1)
    for i = 1, self.nOutputPlane do
        self.gradBias[i] = gradOutput[i]:dot(ones)
    end
    print(input:size(2) - 2*math.floor(self.kH/2))
    for c1star = 1, self.nInputPlane do
        for c2star = 1, self.nOutputPlane do
            for istar = 1, self.kH do
                for jstar = 1, self.kW do
                    for i = 1, gradOutput:size(2) do
                        for j = 1, gradOutput:size(3) do
                                self.gradWeight[c2star][c1star][istar][jstar] = self.gradWeight[c2star][c1star][istar][jstar] + gradOutput[c2star][i][j] * input[c1star][i+istar-1][j+jstar-1]
                        end
                    end
                end
            end
        end
    end
end


    
    
    
    
        
                    
            
    
    
        
