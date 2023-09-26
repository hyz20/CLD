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
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2, Wrap_Dataset_Pairwise4, Wrap_Dataset_tobit, Wrap_Dataset_tobit2
from utils.get_sparse import get_sparse_feature
from utils.early_stop import EarlyStopping
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.loss_func import BPRLoss, HingeLoss,BPRLoss_log_norm, BPRLoss_var_reg, BPRLoss_sel,Dual_BCELoss,Dual_BPRLoss, TobitLoss,TobitLoss2
from models.toy_model import rank_model, rank_model2, rank_model2_2, rank_model_pro, rank_model_tobit

class IPS_tobit2(Learning_Alg):
    def __init__(self, args):
        super(IPS_tobit2, self).__init__(args)
        
    def train(self):
        t0 = time.time()
        self.test_tool, self.in_size = super(IPS_tobit2, self)._load_test_and_get_tool()
        t1 = time.time()
        self.vali_tool = self._load_vali_and_get_tool()
        t2 = time.time()
        input_train_loader = self._load_and_wrap_train()
        t3 = time.time()
        print('load test time:', t1-t0)
        print('load vali time:', t2-t1)
        print('load train time:', t3-t2)
        self.model, self.optim, self.scheduler, self.early_stopping = self._init_train_env()
        self._train_iteration(input_train_loader)
        super(IPS_tobit2, self)._test_and_save()
        
    def _train_one_batch(self, batch):
        #print('------------------------')
        # t0 = time.time()
        self.optim.zero_grad()
        if self.pairwise:
            Tobit_lossfunc = TobitLoss(rho=self.rho, weight=batch[3])
            output_sel_score, output_score =  self.model(batch[0])
            output_sel_score = output_sel_score.view(batch[0].size(0))
            output_score = output_score.view(batch[0].size(0))
            train_loss = Tobit_lossfunc(output_score, batch[1], output_sel_score, batch[2])
        # t1 = time.timke()
        train_loss.backward()
        # t2 = time.time()
        self.optim.step()
        # t3 = time.time()
        # if np.random.rand()>0.95:
        #     print('Time of loss forward:',t1-t0)
        #     print('Time of loss backward:',t2-t1)
        #     print('Time of optimize:',t3-t2)
        #     print('=======================================')
        return train_loss


    def _train_iteration(self, input_train_loader):
        # strat training
        dur = []
        for epoch in range(self.epoch):
            if epoch >= 0:
                t0 = time.time()
            
            loss_log = []
            self.model.train()
            
            tt_2=0
            for _id, batch in enumerate(input_train_loader):
                # tt_1 = time.time()
                # print('Make batch time:',tt_2-tt_1)
                train_loss = self._train_one_batch(batch)
                loss_log.append(train_loss.item())
                # tt_2 = time.time()

            # evaluate performance on validation click log
            # if self.pairwise:
            #     val_loss, eval_result = self.vali_tool.evaluate(self.model, 
            #                                                     loss_type=BPRLoss)
            # else:
            #     val_loss, eval_result = self.vali_tool.evaluate(self.model, 
            #                                                     loss_type=BCELoss)
            # ndcg_val = eval_result['NDCG'][self.topK]
            # mrr_val = eval_result['MRR'][self.topK]
            # precision_val = eval_result['Precision'][self.topK]

            test_result = self.test_tool.evaluate(self.model)
            
            ndcg_tst = test_result['NDCG'][self.topK]
            mrr_tst = test_result['MRR'][self.topK]
            precision_tst = test_result['Precision'][self.topK]
            map_tst = test_result['MAP'][self.topK]
            ndcg_full_tst = test_result['NDCG_full'][self.topK]

            # self.scheduler.step(np.mean(loss_log))
            # if self.schedul_mode == 'max':
            #     self.scheduler.step(ndcg_val)
            # elif self.schedul_mode == 'min':
            #     self.scheduler.step(val_loss)

            # if epoch >= self.epoch_start:
            #     if self.schedul_mode == 'max':
            #         self.early_stopping(ndcg_val*(-1), self.model)
            #     elif self.schedul_mode == 'min': 
            #         self.early_stopping(val_loss, self.model)
                

            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     break 

            torch.save(self.model.state_dict(), '{}_checkpoint.pt'.format(self.fout))
            
            if epoch >= 0:
                dur.append(time.time() - t0)
            

            """
            for name, parms in self.model.named_parameters():
	                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            """
            # print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
            #         "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
            #                                         ndcg_val, mrr_val, precision_val))
            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Test_NDCG@10 {:.4f} | "
                    "Test_MRR@10 {:.4f}| Test_Precision@10 {:.4f} | Test_MAP@10 {:.4f} | Test_NDCG_full {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),ndcg_tst,
                                                    mrr_tst, precision_tst, map_tst, ndcg_full_tst))
                                        

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
            train_log_fe = pd.merge(train_log,train_dat,on=['qid','did'])
            train_log_fe.drop('rankPosition_y',axis=1, inplace=True)
            train_log_fe.drop('label_y',axis=1, inplace=True)
            train_log_fe.drop('ips_weight_for_train',axis=1, inplace=True)
            train_log_fe.rename(columns={'label_x':'label'},inplace = True)
            train_log_fe.rename(columns={'rankPosition_x':'rankPosition'},inplace = True)
            if self.topK > 0:
                train_log_fe['isSelect'] = train_log_fe['rankPosition'].apply(lambda x: 1 if x<self.topK else 0)
            input_train = Wrap_Dataset_tobit2(train_log_fe['isSelect'].to_list(),
                                                    train_log_fe['ips_weight'].to_list(),
                                                    train_log_fe['isClick'].to_list(),
                                                    train_log_fe['feature'].to_list(),
                                                    sparse_tag=self.sparse_tag)
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return input_train_loader


if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = IPS_tobit2(args)
    learner.train()