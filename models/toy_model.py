import torch 
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

class rank_model(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model,self).__init__()
        self.linear_proj = nn.Sequential(
            # nn.Linear(in_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(int(hidden_size/2)),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(hidden_size, 1, bias=False),
            # #nn.Dropout(drop_out),
            # #nn.ELU()

            nn.Linear(in_size, 256),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(64, 1, bias=False),


            # nn.Linear(in_size, 1, bias=True),
            )
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            output = self.linear_proj(input_vec.to_dense())
        else:
            output = self.linear_proj(input_vec)
        #output = nn.ReLU6(logit)
        #prob = torch.sigmoid(logit)
        return output


class rank_model_linear(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model_linear,self).__init__()
        self.linear_proj = nn.Sequential(
                nn.Linear(in_size, 1, bias=True)
            )
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            output = self.linear_proj(input_vec.to_dense())
        else:
            output = self.linear_proj(input_vec)
        #output = nn.ReLU6(logit)
        #prob = torch.sigmoid(logit)
        return output


class rank_model_pro(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model_pro,self).__init__()
        self.transform = nn.Sequential(
            # nn.Linear(in_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU()

            nn.Linear(in_size, 256),
            nn.Dropout(drop_out),
            nn.ELU()
            
            )

        self.linear_proj = nn.Sequential(
            self.transform,

            nn.Linear(256, 128),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(64, 1, bias=False),

            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(int(hidden_size/2)),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            
            # nn.Linear(hidden_size, 1, bias=False)
            )

        self.observe_model = nn.Sequential(
            self.transform,

            nn.Linear(256, 1, bias=False)

            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(int(hidden_size/2)),
            # nn.Dropout(drop_out),
            # nn.ELU(),

            # nn.Linear(hidden_size, 1, bias=False)
            )
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            output = self.linear_proj(input_vec.to_dense())
            output_sel = self.observe_model(input_vec.to_dense())
        else:
            output = self.linear_proj(input_vec)
            output_sel = self.observe_model(input_vec)
        return output_sel, output



class rank_model_tobit(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model_tobit,self).__init__()
        # self.rho_val = torch.full((1,),rho, requires_grad =True).cuda()
        # self.rho_val = torch.Tensor([1],requires_grad =True)
        # self.rho_train = nn.Linear(1, 1, bias=False)
        self.linear_proj = nn.Sequential(
            # nn.Linear(in_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),

            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),

            # nn.Linear(hidden_size, hidden_size),
            # #nn.BatchNorm1d(int(hidden_size/2)),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            
            # nn.Linear(hidden_size, 1, bias=False)

            nn.Linear(in_size, 1, bias=True),
            # nn.Dropout(drop_out)

            # nn.Linear(in_size, 256),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(256, 128),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(128, 64),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(64, 1, bias=False)
            
            )

        self.observe_model = nn.Sequential(
            # nn.Linear(in_size, hidden_size),
            # #nn.BatchNorm1d(hidden_size),
            # nn.Dropout(drop_out),
            # nn.ELU(),
            # nn.Linear(hidden_size, 1, bias=False)

            nn.Linear(in_size, 1, bias=True),
            # nn.Dropout(drop_out)
            )
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            output = self.linear_proj(input_vec.to_dense())
            output_sel = self.observe_model(input_vec.to_dense())
        else:
            output = self.linear_proj(input_vec)
            output_sel = self.observe_model(input_vec)
        # rho = F.tanh(self.rho_val)
        # rho = self.rho_val
        # rho = F.tanh(self.rho_train(self.rho_val))
        return output_sel, output


class rank_model_pro2(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model_pro2,self).__init__()
        # self.transform = nn.Sequential(
        #     nn.Linear(in_size, 256),
        #     nn.Dropout(drop_out),
        #     nn.ELU()
            
        #     )

        self.linear_proj = nn.Sequential(
            # self.transform,
            nn.Linear(in_size, 256),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(64, 1, bias=False),

            )

        self.observe_model = nn.Sequential(
            # self.transform,

            # nn.Linear(256, 1, bias=False)
            nn.Linear(in_size, 1, bias=False)

            )
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            output = self.linear_proj(input_vec.to_dense())
            output_sel = self.observe_model(input_vec.to_dense())
        else:
            output = self.linear_proj(input_vec)
            output_sel = self.observe_model(input_vec)
        return output_sel, output


class rank_model2(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model2,self).__init__()
        self.in_layer = nn.Sequential(nn.Linear(in_size, hidden_size), 
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.hidden_layer1 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.hidden_layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.out_layer = nn.Sequential(nn.Linear(hidden_size, 1, bias=False))
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                # nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            proj_1 = self.in_layer(input_vec.to_dense())
            output_1 = self.out_layer(proj_1)

            proj_2 = self.hidden_layer1(self.in_layer(input_vec.to_dense()))
            output_2 = self.out_layer(proj_2)

            proj_3 = self.hidden_layer2(self.hidden_layer1(self.in_layer(input_vec.to_dense())))
            output_3 = self.out_layer(proj_3)

            all_pred = torch.cat((output_1, output_2, output_3), axis=1)
            var = torch.var(all_pred, axis=1)
        else:
            proj_1 = self.in_layer(input_vec)
            output_1 = self.out_layer(proj_1)

            proj_2 = self.hidden_layer1(self.in_layer(input_vec))
            output_2 = self.out_layer(proj_2)

            proj_3 = self.hidden_layer2(self.hidden_layer1(self.in_layer(input_vec)))
            output_3 = self.out_layer(proj_3)


            all_pred = torch.cat((output_1, output_2, output_3), axis=1)
            var = torch.var(all_pred, axis=1)
        #output = nn.ReLU6(logit)
        #prob = torch.sigmoid(logit)
        # print(all_pred.size())
        # print(var.size())
        # print(output_3.size())
        #print(var.size()==output_3.size())
        return var, output_3

class rank_model2_2(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model2_2,self).__init__()
        self.model0 = nn.Sequential(nn.Linear(in_size, 1))
        self.model1 = nn.Sequential(nn.Linear(in_size, hidden_size), 
                                    nn.Dropout(drop_out), 
                                    nn.ELU(),
                                    nn.Linear(hidden_size, 1, bias=False))
        self.model2 = nn.Sequential(nn.Linear(in_size, hidden_size), 
                                    nn.Dropout(drop_out), 
                                    nn.ELU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU(),
                                    nn.Linear(hidden_size, 1, bias=False))
        self.model3 = nn.Sequential(nn.Linear(in_size, hidden_size), 
                                    nn.Dropout(drop_out), 
                                    nn.ELU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU(),
                                    nn.Linear(hidden_size, 1, bias=False))

    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                # nn.init.kaiming_normal_(op.weight)

    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            output_0 = self.model0(input_vec.to_dense())
            output_1 = self.model1(input_vec.to_dense())
            output_2 = self.model2(input_vec.to_dense())
            output_3 = self.model3(input_vec.to_dense())
            with torch.no_grad():
                all_pred = torch.cat((output_0, output_1, output_2, output_3), axis=1)
                var = torch.var(all_pred, axis=1)
        else:
            output_0 = self.model0(input_vec)
            output_1 = self.model1(input_vec)
            output_2 = self.model2(input_vec)
            output_3 = self.model3(input_vec)

            with torch.no_grad():
                all_pred = torch.cat((output_0, output_1, output_2, output_3), axis=1)
                var = torch.var(all_pred, axis=1)

        return var, output_3, output_2, output_1, output_0
        # return torch.tensor([0]).cuda(), output_3, torch.tensor([0]).cuda(), torch.tensor([0]).cuda(), torch.tensor([0]).cuda()


class rank_model2_tobit(Module):
    def __init__(self, in_size, hidden_size, drop_out):
        super(rank_model2_tobit,self).__init__()
        self.in_layer = nn.Sequential(nn.Linear(in_size, hidden_size), 
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.hidden_layer1 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.hidden_layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.out_layer = nn.Linear(hidden_size, 1, bias=False)

        self.in_select_layer = nn.Sequential(nn.Linear(in_size, hidden_size), 
                                    nn.Dropout(drop_out), 
                                    nn.ELU())
        self.out_select_layer = nn.Linear(hidden_size, 1, bias=False)
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)

    
    def forward(self, input_vec):
        if input_vec.layout == torch.sparse_coo:
            # print(input_vec.shape)
            proj_1 = self.in_layer(input_vec.to_dense())
            output_1 = self.out_layer(proj_1)

            proj_2 = self.hidden_layer1(self.in_layer(input_vec.to_dense()))
            output_2 = self.out_layer(proj_2)

            proj_3 = self.hidden_layer2(self.hidden_layer1(self.in_layer(input_vec.to_dense())))
            output_3 = self.out_layer(proj_3)

            proj_select = self.in_select_layer(input_vec.to_dense())
            output_select = self.out_select_layer(proj_select)
            # output_select = self.out_select_layer(proj_3)

            all_pred = torch.cat((output_1, output_2, output_3), axis=1)
            var = torch.var(all_pred, axis=1)
        else:
            proj_1 = self.in_layer(input_vec)
            output_1 = self.out_layer(proj_1)

            proj_2 = self.hidden_layer1(self.in_layer(input_vec))
            output_2 = self.out_layer(proj_2)

            proj_3 = self.hidden_layer2(self.hidden_layer1(self.in_layer(input_vec)))
            output_3 = self.out_layer(proj_3)

            proj_select = self.in_select_layer(input_vec)
            output_select = self.out_select_layer(proj_select)
            # output_select = self.out_select_layer(proj_3)

            all_pred = torch.cat((output_1, output_2, output_3), axis=1)
            var = torch.var(all_pred, axis=1)
        #output = nn.ReLU6(logit)
        #prob = torch.sigmoid(logit)
        # print(all_pred.size())
        # print(var.size())
        # print(output_3.size())
        #print(var.size()==output_3.size())
        return var, output_3, output_select


class rank_model3(Module):
    def __init__(self, in_size, hidden_size, drop_out, topK):
        super(rank_model3,self).__init__()
        self.propensity_model = nn.Sequential(nn.Linear(topK, 1), 
                                    nn.Dropout(drop_out), 
                                    nn.ReLU())
        
        self.linear_proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            #nn.BatchNorm1d(int(hidden_size/2)),
            nn.Dropout(drop_out),
            nn.ELU(),
            nn.Linear(int(hidden_size/2), 1, bias=False),
            #nn.BatchNorm1d(),
            #nn.Dropout(drop_out),
            nn.ELU()
            )
        
    def weight_init(self):
         for op in self.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_normal_(op.weight)
                #nn.init.constant_(op.bias, 0.0)
                #nn.init.kaiming_normal_(op.weight)
    
    def forward(self, input_vec, input_position):
        if input_vec.layout == torch.sparse_coo:
            rel_pred = self.linear_proj(input_vec.to_dense())
            propensity_pred = self.propensity_model(input_position.to_dense())
        else:
            rel_pred = self.linear_proj(input_vec)
            propensity_pred = self.propensity_model(input_position)
        
        final_pred = rel_pred * propensity_pred
        
        return final_pred
        
        

if __name__ =="__main__":
    model = rank_model(46, 16, 0.5)
    model.weight_init()