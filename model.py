import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_DIM = 6

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(512, 256)
        #self.drop1 = torch.nn.Dropout(0.5)
        self.lin2 = torch.nn.Linear(256, 128)
        #self.drop2 = torch.nn.Dropout(0.4)
        self.lin3 = torch.nn.Linear(128, 64)
        #self.drop3 = torch.nn.Dropout(0.3)
        self.lin4 = torch.nn.Linear(64, 32)
        self.lin5 = torch.nn.Linear(32, 16)
        self.lin6 = torch.nn.Linear(16, SMALL_DIM)

        self.act = torch.nn.Tanh()


    def forward(self, x):
        result = self.lin1(x)
        result = self.act(result)
        #result = self.drop1(result)
        result = self.lin2(result)
        result = self.act(result)
        #result = self.drop2(result)
        result = self.lin3(result)
        result = self.act(result)
        #result = self.drop3(result)
        result = self.lin4(result)
        result = self.act(result)
        result = self.lin5(result)
        result = self.act(result)
        result = self.lin6(result)
        result = self.act(result)

        result = result / result.norm(dim=-1, keepdim=True)
        return result
    
class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(SMALL_DIM, 16)
        #self.drop1 = torch.nn.Dropout(0.5)
        self.lin2 = torch.nn.Linear(16, 32)
        #self.drop2 = torch.nn.Dropout(0.4)
        self.lin3 = torch.nn.Linear(32, 64)
        #self.drop3 = torch.nn.Dropout(0.3)
        self.lin4 = torch.nn.Linear(64, 128)
        self.lin5 = torch.nn.Linear(128, 256)
        self.lin6 = torch.nn.Linear(256, 256)
        self.lin7 = torch.nn.Linear(256, 256)
        self.lin8 = torch.nn.Linear(256, 512)
        
        self.act = torch.nn.Tanh()

    def forward(self, x):
        result = self.lin1(x)
        result = self.act(result)
        #result = self.drop1(result)
        result = self.lin2(result)
        result = self.act(result)
        #result = self.drop2(result)
        result = self.lin3(result)
        result = self.act(result)
        #result = self.drop3(result)
        result = self.lin4(result)
        result = self.act(result)
        result = self.lin5(result)
        result = self.act(result)
        result = self.lin6(result)
        result = self.act(result)
        result = self.lin7(result)
        result = self.act(result)
        result = self.lin8(result)
        result = self.act(result)

        result = result / result.norm(dim=-1, keepdim=True)
        return result