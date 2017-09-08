
local DeformableConvolution, parent = torch.class('nn.DeformableConvolution', 'nn.Module')

function DeformableConvolution:__init(nInputPlane, nOutputPlane, kW, kH, padW, padH)
   parent.__init(self)


   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   
   self.padW = padW or 0
   self.padH = padH or self.padW    

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

end

function DeformableConvolution:updateOutput(input)
    local wOutputImage = input:size(3)-2*math.floor(self.kW/2)+2*self.padW
    local hOutputImage = input:size(2)-2*math.floor(self.kH/2)+2*self.padH
    
    self.output = torch.Tensor(self.nOutputPlane, hOutputImage, wOutputImage)
    for i=1,self.nOutputPlane do
        for j=1,self.nInputPlane do
            self.output[i] = self.output[i]+self:convolution(self.weight[i][j],input[j], wOutputImage, hOutputImage)
        end
        self.output[i] = self.output[i]+self.bias[i]
    end
    return self.output
end
    
    

function DeformableConvolution:convolution(kernel, image, wOutputImage, hOutputImage)
    
    output_image = torch.Tensor(hOutputImage,wOutputImage):zero()
    local image_framed = torch.Tensor(image:size(1)+2*self.padH,image:size(2)+2*self.padW):zero()
    
    for i = 1,image:size(1) do
        for j=1,image:size(2) do
            image_framed[i+self.padH][j+self.padW] = image[i][j]
        end
    end
    
    for i=1,wOutputImage do
        for j=1,hOutputImage do
            for k=0,self.kW-1 do
                for l=0,self.kH-1 do
                    output_image[j][i] = output_image[j][i] + image_framed[j+l][i+k]*kernel[l+1][k+1]
                end
            end
        end
    end
    return output_image
end

function DeformableConvolution:rotateAbout180(input)
    output = torch.Tensor(input:size())
    for i = 1, input:size(1) do
        for j = 1, input:size(2) do
            for k = 1, input:size(3) do 
                output[i][j][k] = input[i][input:size(2)-j+1][input:size(3)-k+1]
            end
        end
    end
    return output
end
        


function DeformableConvolution:updateGradInput(input,gradOutput)
    self.gradInput = torch.Tensor(input:size()):zero()
    for c1star = 1,self.nInputPlane do
        for istar = 1, input:size(2) do
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
    
    
function DeformableConvolution:accGradParameters(input,gradOutput, scale)
    self.gradWeight:zero()
    self.gradBias:zero()
    ones = torch.Tensor(gradOutput:size(2),gradOutput:size(3)):fill(1)
    for i = 1, self.nOutputPlane do
        self.gradBias[i] = gradOutput[i]:dot(ones)
    end
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


    
    
    
    
        
                    
            
    
    
        
