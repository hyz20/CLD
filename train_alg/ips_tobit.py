import torch
import pandas as pd
import numpy as np
from torch.nn import BCELoss,BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import time
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2, Wrap_Dataset_Pairwise4, Wrap_Dataset_tobit
from utils.get_sparse import get_sparse_feature
from utils.early_stop import EarlyStopping
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.loss_func import BPRLoss, HingeLoss,BPRLoss_log_norm, BPRLoss_var_reg, BPRLoss_sel,Dual_BCELoss,Dual_BPRLoss, TobitLoss
from models.toy_model import rank_model, rank_model2, rank_model2_2, rank_model_pro, rank_model_tobit

class IPS_tobit(Learning_Alg):
    def __init__(self, args):
        super(IPS_tobit, self).__init__(args)
        
    def train(self):
        t0 = time.time()
        self.test_tool, self.in_size = super(IPS_tobit, self)._load_test_and_get_tool()
        t1 = time.time()
        self.vali_tool = self._load_vali_and_get_tool()
        t2 = time.time()
        input_train_loader = self._load_and_wrap_train()
        t3 = time.time()
        print('load test time:', t1-t0)
        print('load vali time:', t2-t1)
        print('load train time:', t3-t2)
        self.model, self.optim, self.scheduler, self.early_stopping = self._init_train_env()
        super(IPS_tobit, self)._train_iteration(input_train_loader)
        super(IPS_tobit, self)._test_and_save()
        
    def _train_one_batch(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            Tobit_lossfunc = TobitLoss(rho=self.rho, weight=batch[3])
            output_sel_score, output_score =  self.model(batch[0])
            output_sel_score = output_sel_score.view(batch[0].size(0))
            output_score = output_score.view(batch[0].size(0))
            train_loss = Tobit_lossfunc(output_score, batch[1], output_sel_score, batch[2])

        train_loss.backward()
        self.optim.step()
        return train_loss


    def _init_train_env(self):
        model = rank_model_tobit(self.in_size, self.hidden_size, self.drop_out)
        # print('))))))))))))))')
        # model = rank_model(self.in_size, self.hidden_size, self.drop_out)
        model.weight_init()
        if self.use_cuda:
            #model = nn.DataParallel(model)
            model = model.cuda()
        
        if self.optimizer == 'adam':
            optim = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optim = SGD(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer== 'adadelta':
            optim = Adadelta(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optim, 
                                      patience=10, 
                                      mode=self.schedul_mode,
                                      threshold=1e-6,
                                      verbose=True)
        early_stopping = EarlyStopping(self.fout, self.patience, verbose=True) 
        return model, optim, scheduler, early_stopping

        
    def _load_vali_and_get_tool(self):
        if self.pairwise:
            vali_log = pd.read_json(self.fin + 'click_log/Vali_log.json')
            vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair.json')
            if self.topK > 0:
                vali_log = vali_log[vali_log['rankPosition'] < self.topK]
                vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair_topk.json')
                # vali_log_pair = vali_log_pair[vali_log_pair['tag']==1]
                vali_log_pair.reset_index(drop=True, inplace=True)
            #self.train_alg='naive'
            vali_tool = Vali_Evaluator(vali_log, 
                                       self.eval_positions, 
                                       use_cuda=self.use_cuda, 
                                       with_weight=self.train_alg, 
                                       pair_wise=True, pair_df=vali_log_pair)
        else:
            vali_log = pd.read_json(self.fin + 'click_log/Vali_log.json')
            if self.topK > 0:
                vali_log = vali_log[vali_log['rankPosition'] < self.topK]
            vali_tool = Vali_Evaluator(vali_log, 
                                       self.eval_positions, 
                                       use_cuda=self.use_cuda, 
                                       with_weight=self.train_alg)
        return vali_tool
    
    def _load_and_wrap_train(self):
        if self.pairwise:
            # train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair.json')
            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            train_dat = pd.read_json(self.fin + 'json_file/Train.json')
            train_dat = format_trans(train_dat)
            if self.topK > 0:
                train_log['isSelect'] = train_log['rankPosition'].apply(lambda x: 1 if x<self.topK else 0)
            input_train = Wrap_Dataset_tobit(train_log['did'].to_list(), 
                                                    train_log['isSelect'].to_list(),
                                                    train_log['ips_weight'].to_list(),
                                                    train_log['isClick'].to_list(),
                                                    train_dat,
                                                    sparse_tag=self.sparse_tag)
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return input_train_loader


if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = IPS_tobit(args)
    learner.train()