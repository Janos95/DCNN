
local DeformableConv, parent = torch.class('nn.DeformableConv', 'nn.Module')

function DeformableConv:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)


   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   
   self.padW = math.floor(self.kW/2)
   self.padH = math.floor(self.kH/2)



   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kW, kH)
   self.bias = torch.Tensor(nOutputPlane)

end

function DeformableConv:updateOutput(input)
    self.output = torch.Tensor(self.nOutputPlane, input:size(2),input:size(3))
    for i=1,self.nOutputPlane do
        for j=1,self.nInputPlane do
            self.output[i] = self.output[i]+self:convolution(self.weight[i][j],input[j])
        end
        self.output[i] = self.output[i]+self.bias[i]
    end
    return self.output
end

    
    

function DeformableConv:convolution(kernel, image)
    output_image = torch.Tensor(image:size()):zero()
    local w = image:size(1)
    local h = image:size(2)

    image_framed = torch.Tensor(w+2*self.padW,h+2*self.padH):zero()
    for i = 1,w do
        for j=1,h do
            image_framed[i+self.padW][j+self.padH] = image[i][j]
        end
    end
    
    for i=1,w do
        for j=1,h do
            for k=0,self.kW-1 do
                for l=0,self.kH-1 do
                    output_image[i][j] = output_image[i][j] + image_framed[i+k][j+l]*kernel[k+1][l+1]
                end
            end
        end
    end
    return output_image
end

    
    
        
                    
            
    
    
        
