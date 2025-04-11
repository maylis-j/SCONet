import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, w = 1):
        super().__init__()
        self.w = w

    def forward(self, output, target, deep_supervision=False):
        if not deep_supervision:
            output = torch.swapaxes(output,1,2)
            target = torch.argmax(target,dim=2)
        loss = F.cross_entropy(
                output, target, reduction='none')
        return self.w * loss


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma  

    def forward(self, output, target):        
        output = torch.swapaxes(output,1,2)
        target = torch.argmax(target,dim=2)
        loss = F.cross_entropy(output, target, reduction='none')
        probs = torch.exp(-loss)
        return loss * torch.pow(1.0 - probs, self.gamma)
    

class DiceLoss(nn.Module):
    def __init__(self, epsilon = 1e-5, w = 1):
        super().__init__()
        self.epsilon = epsilon
        self.w = w

    def forward(self, output, target):
        output = F.softmax(output, dim=2)
        
        intersection = output * target
        intersection = intersection.sum(dim=(0,1))

        denominator = output.sum(dim=(0,1)) + target.sum(dim=(0,1))
        denominator = denominator.clamp(min=self.epsilon)
        
        dice = 2*(intersection)/(denominator)
        return self.w * (1 - dice)
       

class CEDiceLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.ce = CELoss(w=alpha)
        self.dice = DiceLoss(w=beta)

    def forward(self, output, target):
        ce = self.ce(output, target).mean()
        dice = self.dice(output, target).mean()
        #print(f"\tCE loss : {ce.item():.4f}, Dice loss : {dice.item():.4f}")
        return ce + dice   


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta       
        self.fl = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, output, target):
        fl = self.fl(output, target).mean()
        dice = self.dice(output, target).mean()
        #print(f"\tFocal loss : {fl.item():.4f}, Dice loss : {dice.item():.4f}")
        return self.alpha * fl + self.beta * dice
    

def _create_loss(name, params):
    if name == 'DiceLoss':
        return DiceLoss()
    elif name == 'FocalLoss':
        gamma = params['gamma']
        return FocalLoss(gamma)                 
    elif name == 'CELoss':
        alpha = params['alpha']    
        return CELoss(w=alpha)  
    elif name == 'CEDiceLoss':
        alpha = params['alpha']
        beta = params['beta']        
        return CEDiceLoss(alpha, beta)
    elif name == 'FocalDiceLoss':
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        return FocalDiceLoss(alpha, beta, gamma)    
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
