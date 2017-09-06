
local DeformableConvolution, parent = torch.class('nn.DeformableConvolution', 'nn.Module')

function DeformableConvolution:__init(nInputPlane, nOutputPlane, kW, kH, padW, padH)
   parent.__init(self)


   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   
   self.padW = padW or 0
   self.padH = padH or self.padW    

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kW, kH)
   self.bias = torch.Tensor(nOutputPlane)

end

function DeformableConvolution:updateOutput(input)
    local wOutputImage = input:size(2)-2*math.floor(self.kW/2)+2*self.padW
    local hOutputImage = input:size(3)-2*math.floor(self.kH/2)+2*self.padH
    
    self.output = torch.Tensor(self.nOutputPlane, wOutputImage, hOutputImage)
    for i=1,self.nOutputPlane do
        for j=1,self.nInputPlane do
            self.output[i] = self.output[i]+self:convolution(self.weight[i][j],input[j], wOutputImage, hOutputImage)
        end
        self.output[i] = self.output[i]+self.bias[i]
    end
    return self.output
end
    
    

function DeformableConvolution:convolution(kernel, image, wOutputImage, hOutputImage)
    
    output_image = torch.Tensor(wOutputImage,hOutputImage):zero()
    local image_framed = torch.Tensor(image:size(1)+2*self.padW,image:size(2)+2*self.padH):zero()
    
    for i = 1,image:size(1) do
        for j=1,image:size(2) do
            image_framed[i+self.padW][j+self.padH] = image[i][j]
        end
    end
    
    for i=1,wOutputImage do
        for j=1,hOutputImage do
            for k=0,self.kW-1 do
                for l=0,self.kH-1 do
                    output_image[i][j] = output_image[i][j] + image_framed[i+k][j+l]*kernel[k+1][l+1]
                end
            end
        end
    end
    return output_image
end

function DeformableConv:rotate(input)
    output = torch.Tensor(input:size())
    for i = 1, input:size(1) do
        for j = 1, input:size(2) do
            for k = 1, input:size(3) do 
                output[i][j][k] = input[i][input:size(2)-i+1][input:size(3)-j+1]
            end
        end
    end
    return output
end
        


function DeformableConv:updateGradInput(input,gradOutput)
    
    

    
    
        
                    
            
    
    
        
