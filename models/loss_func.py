import torch
import numpy as np
import time
from torch import nn
from torch.nn import MarginRankingLoss
from torch.nn import functional as F

class Unbias_MSELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, score, target):
        return (torch.square(score) - 2 * torch.mul(target, score)).mean()


class Unbias_MSELoss_reg(nn.Module):
    def __init__(self, weight=None, C=5):
        super().__init__()
        self.weight = weight
        self.C = C

    def forward(self, var, score, target):
        #all_pred = torch.cat((score1.unsqueeze(0), score2.unsqueeze(0), score3.unsqueeze(0)), axis=0)
        if self.weight is not None:
            return (torch.square(score) - 2 * torch.mul(torch.mul(target,self.weight), score) + torch.mul(self.C, var)).mean()
        else:
            return (torch.square(score) - 2 * torch.mul(target, score) + torch.mul(self.C, var)).mean()


class Tobit_MLELoss(nn.Module):
    def __init__(self, weight=None, C=5, rho=0.2):
        super().__init__()
        self.weight = weight
        self.C = C
        self.rho = torch.Tensor([rho]).cuda()

    def forward(self, var_r, score_r, score_s, target_r, target_s):
        if self.weight is not None:
            val_mle = torch.square(score_r) - 2 * torch.mul(torch.mul(target_r,self.weight), score_r)
            val_var = torch.mul(self.C, var_r)
            val_unsel = torch.log(1 - F.sigmoid(score_s) + 1e-8)

            numerator_part = score_s + torch.mul(self.rho, torch.mul(target_r,self.weight) - score_r)
            denominator_part = torch.sqrt(1 - torch.pow(self.rho,2))
            val_sel = F.logsigmoid(torch.div(numerator_part, denominator_part))
        else:
            val_mle = torch.square(score_r) - 2 * torch.mul(target_r, score_r)
            val_var = torch.mul(self.C, var_r)
            val_unsel = torch.log(1 - F.sigmoid(score_s) + 1e-8)

            numerator_part = score_s + torch.mul(self.rho, target_r - score_r)
            denominator_part = torch.sqrt(1 - torch.pow(self.rho,2))
            val_sel = F.logsigmoid(torch.div(numerator_part, denominator_part))

        return (torch.mul(target_s, val_mle) + torch.mul(target_s, val_var) - torch.mul(target_s, val_sel) - torch.mul(1 - target_s, val_unsel)).mean()
    

class Tobit_MLELoss2(nn.Module):
    def __init__(self, weight=None, C=5, rho=0.2):
        super().__init__()
        self.weight = weight
        self.C = C
        self.rho = torch.Tensor([rho]).cuda()

    def forward(self, var_r, score_r, score_s, target_r, target_s):
        if self.weight is not None:
            val_mle = torch.square(score_r) - 2 * torch.mul(torch.mul(target_r,self.weight), score_r)
            val_var = torch.mul(self.C, var_r)
            val_unsel = torch.log(1 - F.sigmoid(score_s) + 1e-8)

        else:
            val_mle = torch.square(score_r) - 2 * torch.mul(target_r, score_r)
            val_var = torch.mul(self.C, var_r)
            val_unsel = torch.log(1 - F.sigmoid(score_s) + 1e-8)

        return (torch.mul(target_s, val_mle) + torch.mul(target_s, val_var) - torch.mul(1 - target_s, val_unsel)).mean()


class BPRLoss_log_norm_reg_tobit(nn.Module):
    def __init__(self, weight=None, C=5, rho=0.0):
        super().__init__()
        self.weight = weight
        self.C = C
        self.rho = torch.Tensor([rho]).cuda()

    def forward(self, pos_sel, neg_sel, pos_var, neg_var, pos_score, neg_score):
        #all_pred = torch.cat((score1.unsqueeze(0), score2.unsqueeze(0), score3.unsqueeze(0)), axis=0)
        if self.weight is not None:
            #self.weight = torch.log(torch.Tensor([1]).cuda()+self.weight) 
            #bpr_val = torch.mul(torch.div(self.weight,self.weight.mean()), -F.logsigmoid(pos_score - neg_score)).mean()
            bpr_val = torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
            var_val = (torch.mul(self.C, pos_var) + torch.mul(self.C, neg_var)).mean()

            numerator_part = pos_sel + neg_sel + torch.mul(self.rho, torch.mul(pos_score - neg_score, self.weight))
            denominator_part = torch.sqrt(1 - torch.pow(self.rho,2))
            sel_val = -F.logsigmoid(torch.div(numerator_part, denominator_part))
        else:
            bpr_val = -F.logsigmoid(pos_score - neg_score).mean()
            var_val = (torch.mul(self.C, pos_var) + torch.mul(self.C, neg_var)).mean()

            numerator_part = pos_sel + neg_sel + torch.mul(self.rho, pos_score - neg_score)
            denominator_part = torch.sqrt(1 - torch.pow(self.rho,2))
            sel_val = -F.logsigmoid(torch.div(numerator_part, denominator_part))
            
        return bpr_val + var_val + sel_val


class BPRLoss_var_reg(nn.Module):
    def __init__(self, weight=None,C=5):
        super().__init__()
        self.weight = weight
        self.C = C

    def forward(self, pos_var, neg_var, pos_score, neg_score):
        #all_pred = torch.cat((score1.unsqueeze(0), score2.unsqueeze(0), score3.unsqueeze(0)), axis=0)
        if self.weight is not None:
            # self.weight = torch.log(torch.Tensor([1]).cuda()+self.weight) 
            # bpr_val = torch.mul(torch.div(self.weight,self.weight.mean()), -F.logsigmoid(pos_score - neg_score)).mean()
            bpr_val = torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
            var_val = (torch.mul(self.C, pos_var) + torch.mul(self.C, neg_var)).mean()
            
        else:
            bpr_val = -F.logsigmoid(pos_score - neg_score).mean()
            var_val = (torch.mul(self.C, pos_var) + torch.mul(self.C, neg_var)).mean()
            
        return bpr_val + var_val


class BPRLoss_log_norm(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, pos_score, neg_score):
        if self.weight != None:
            self.weight = torch.log(torch.Tensor([1]).cuda()+self.weight) 
            #print(self.weight.mean())
            #print(self.weight.mean())
            return torch.mul(torch.div(self.weight,self.weight.mean()), -F.logsigmoid(pos_score - neg_score)).mean()
            # return torch.div(torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).sum(), torch.sum(self.weight))
            # return torch.mul(torch.div(self.weight,torch.Tensor([1.332234]).cuda()), -F.logsigmoid(pos_score - neg_score)).mean()
            # return torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
        else:
            return -F.logsigmoid(pos_score - neg_score).mean()


class BPRLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, pos_score, neg_score):
        if self.weight != None:
            # self.weight = torch.log(torch.Tensor([1]).cuda()+self.weight) 
            # return torch.mul(torch.div(self.weight, torch.sum(self.weight)), -F.logsigmoid(pos_score - neg_score)).mean()
            # return torch.div(torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).sum(), torch.sum(self.weight))
            # return torch.mul(torch.div(self.weight,self.weight.mean()), -F.logsigmoid(pos_score - neg_score)).mean()
            return torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
        else:
            return -F.logsigmoid(pos_score - neg_score).mean()


class TobitLoss(nn.Module):
    def __init__(self,rho, weight=None, C=1):
        super().__init__()
        self.weight = weight
        self.C = C
        self.rho = torch.FloatTensor([rho]).cuda()

    def forward(self, score, r_tag, sel_score, sel_tag):
        mle_val = torch.square(torch.mul(self.weight, r_tag) - score)
        conditional_bce = -F.logsigmoid(torch.div(sel_score + torch.mul(self.rho,torch.mul(self.weight,r_tag)-score), torch.sqrt(1 - torch.pow(self.rho,2))))
        unsel_val = -torch.log(1 - F.sigmoid(sel_score) + 1e-8)
        return (torch.mul(sel_tag, conditional_bce + mle_val) + torch.mul(1-sel_tag, unsel_val)).mean()


class TobitLoss2(nn.Module):
    def __init__(self,rho, weight=None, C=1):
        super().__init__()
        self.weight = weight
        self.C = C
        self.rho = torch.FloatTensor([rho]).cuda()

    def forward(self, score, r_tag, sel_score, sel_tag):
        #t1 = time.time()
        mle_val = torch.square(torch.mul(self.weight, r_tag) - score)
        #t2 = time.time()
        conditional_bce = -F.logsigmoid(torch.div(sel_score + torch.mul(self.rho,torch.mul(self.weight,r_tag)-score), torch.sqrt(1 - torch.pow(self.rho,2))))
        #t3 = time.time()
        unsel_val = -torch.log(1 - F.sigmoid(sel_score) + 1e-8)
        #t4 = time.time()
        # print('Time of mle loss:',t2-t1)
        # print('Time of conditional bce loss:',t3-t2)
        # print('Time of unsel bce loss:',t4-t3)
        # print('=======================================')
        return (torch.mul(sel_tag, conditional_bce + mle_val) + torch.mul(1-sel_tag, unsel_val)).mean()


class BPRLoss_sel(nn.Module):
    def __init__(self, weight=None, C=1):
        super().__init__()
        self.weight = weight
        self.C = C
        
    def forward(self, pos_score, neg_score, pos_sel_score, neg_sel_score, pos_sel_tag, neg_sel_tag):
        if self.weight != None:
            bpr_val = torch.mul(pos_sel_tag * neg_sel_tag * self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
            pos_sel = F.binary_cross_entropy_with_logits(pos_sel_score, pos_sel_tag)
            neg_sel = F.binary_cross_entropy_with_logits(neg_sel_score, neg_sel_tag)
            return  bpr_val + torch.mul(self.C, pos_sel) + torch.mul(self.C,neg_sel)
        else:
            bpr_val = torch.mul(pos_sel_tag*neg_sel_tag, -F.logsigmoid(pos_score - neg_score)).mean()
            pos_sel = F.binary_cross_entropy_with_logits(pos_sel_score, pos_sel_tag)
            neg_sel = F.binary_cross_entropy_with_logits(neg_sel_score, neg_sel_tag)

            return  bpr_val + torch.mul(self.C, pos_sel) + torch.mul(self.C,neg_sel)


class BPRLoss_sel2(nn.Module):
    def __init__(self, weight=None, C=1):
        super().__init__()
        self.weight = weight
        self.C = C
        
    def forward(self, pos_score, neg_score, pos_sel_score, neg_sel_score, pos_sel_tag, neg_sel_tag):
        if self.weight != None:
            bpr_val = torch.mul(pos_sel_tag * neg_sel_tag * self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
            pos_sel = F.binary_cross_entropy_with_logits(pos_sel_score + (pos_score - neg_score), pos_sel_tag)
            neg_sel = F.binary_cross_entropy_with_logits(neg_sel_score + (pos_score - neg_score), neg_sel_tag)
            return  bpr_val + torch.mul(self.C, pos_sel) + torch.mul(self.C, neg_sel)
        else:
            bpr_val = torch.mul(pos_sel_tag * neg_sel_tag, -F.logsigmoid(pos_score - neg_score)).mean()
            pos_sel = F.binary_cross_entropy_with_logits(pos_sel_score + (pos_score - neg_score), pos_sel_tag)
            neg_sel = F.binary_cross_entropy_with_logits(neg_sel_score + (pos_score - neg_score), neg_sel_tag)

            return  bpr_val + torch.mul(self.C, pos_sel) + torch.mul(self.C, neg_sel)


class BPRLoss_sel3(nn.Module):
    def __init__(self, weight=None, C=1):
        super().__init__()
        self.weight = weight
        self.C = C
        
    def forward(self, pos_score, neg_score, pos_sel_score, neg_sel_score, pos_sel_tag, neg_sel_tag):
        if self.weight != None:
            bpr_val = torch.mul(pos_sel_tag * neg_sel_tag * self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
            pos_sel = (-pos_sel_tag*F.logsigmoid(pos_sel_score + (pos_score - neg_score))- (1-pos_sel_tag)*torch.log(1 - F.sigmoid(pos_sel_score) + 1e-8)).mean()
            neg_sel = (-neg_sel_tag*F.logsigmoid(neg_sel_score + (pos_score - neg_score))- (1-neg_sel_tag)*torch.log(1 - F.sigmoid(neg_sel_score) + 1e-8)).mean()
            return  bpr_val + torch.mul(self.C, pos_sel) + torch.mul(self.C, neg_sel)
        else:
            bpr_val = torch.mul(pos_sel_tag * neg_sel_tag, -F.logsigmoid(pos_score - neg_score)).mean()
            pos_sel = (-pos_sel_tag*F.logsigmoid(pos_sel_score + (pos_score - neg_score))- (1-pos_sel_tag)*torch.log(1 - F.sigmoid(pos_sel_score) + 1e-8)).mean()
            neg_sel = (-neg_sel_tag*F.logsigmoid(neg_sel_score + (pos_score - neg_score))- (1-neg_sel_tag)*torch.log(1 - F.sigmoid(neg_sel_score) + 1e-8)).mean()

            return  bpr_val + torch.mul(self.C, pos_sel) + torch.mul(self.C, neg_sel)


class Dual_BPRLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pos_score, neg_score, pos_sel_tag, neg_sel_tag):
        if self.weight != None:
            # self.weight = torch.log(torch.Tensor([1]).cuda()+self.weight) 
            # return torch.mul(torch.div(self.weight, torch.sum(self.weight)), -F.logsigmoid(pos_score - neg_score)).mean()
            # return torch.div(torch.mul(self.weight, -F.logsigmoid(pos_score - neg_score)).sum(), torch.sum(self.weight))
            # return torch.mul(torch.div(self.weight,self.weight.mean()), -F.logsigmoid(pos_score - neg_score)).mean()
            return torch.mul(pos_sel_tag * neg_sel_tag * self.weight, -F.logsigmoid(pos_score - neg_score)).mean()
        else:
            return torch.mul(pos_sel_tag * neg_sel_tag ,-F.logsigmoid(pos_score - neg_score)).mean()

class Dual_BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pos_sel_score, neg_sel_score, pos_sel_tag, neg_sel_tag):
        pos_sel = F.binary_cross_entropy_with_logits(pos_sel_score, pos_sel_tag)
        neg_sel = F.binary_cross_entropy_with_logits(neg_sel_score, neg_sel_tag)
        return pos_sel+neg_sel


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, weight=None):
        super().__init__()
        self.margin = float(margin)
        self.weight = weight
    
    def forward(self, pos_score, neg_score):
        target = torch.LongTensor([1])
        if self.weight != None:
            return torch.mul(self.weight, F.margin_ranking_loss(pos_score, neg_score, target, margin= self.margin, reduction='none')).mean()
        else:
            return F.margin_ranking_loss(pos_score, neg_score, target, margin= self.margin)


if __name__=="__main__":
    weight = torch.Tensor([1.0, 2.0])

    #bpr_loss = BPRLoss()
    #hinge_loss = HingeLoss(margin=1)
    bpr_loss = BPRLoss(weight=weight)
    hinge_loss = HingeLoss(margin=1,weight=weight)

    x1 = torch.Tensor([4.3, 3.3])
    x2 = torch.Tensor([0.9, 2.9])

    print('bpr loss: ',bpr_loss(x1, x2))
    print('hinge loss: ',hinge_loss(x1, x2))

