local almostIdentity, parent = torch.class('nn.almostIdentity', 'nn.Module')

function almostIdentity:__init()
    parent.__init(self)
    self.i = 0
end

function almostIdentity:updateOutput(input)
   self.output = input
   return self.output
end


function almostIdentity:updateGradInput(input, gradOutput)
    self.i = self.i+1
    if(self.i % 100 ==  0) then
        print(self.i)
    end
    self.gradInput = gradOutput
    --print(self.gradInput)
    return self.gradInput
end

function almostIdentity:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif torch.type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   return self
end
