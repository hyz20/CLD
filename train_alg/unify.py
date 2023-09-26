import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCELoss,BCEWithLogitsLoss,MSELoss
import pandas as pd
import numpy as np
import sys
import time
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise3
from utils.get_sparse import get_sparse_feature
from models.loss_func import BPRLoss, HingeLoss, Unbias_MSELoss,Unbias_MSELoss_reg
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.toy_model import rank_model,rank_model3
from utils.early_stop import EarlyStopping



class Unify(Learning_Alg):
    def __init__(self, args):
        super(Unify, self).__init__(args)
        
    def train(self):
        t0 = time.time()
        self.test_tool, self.in_size = super(Unify, self)._load_test_and_get_tool()
        print(self.in_size)
        t1 = time.time()
        self.vali_tool = self._load_vali_and_get_tool()
        t2 = time.time()
        input_train_loader = self._load_and_wrap_train()
        t3 = time.time()
        print('load test time:', t1-t0)
        print('load vali time:', t2-t1)
        print('load train time:', t3-t2)

        self.model, self.optim, self.scheduler, self.early_stopping = self._init_train_env()
        super(Unify, self)._train_iteration(input_train_loader)
        super(Unify, self)._test_and_save()
    
        
    def _load_vali_and_get_tool(self):
        if self.pairwise:
            vali_log = pd.read_json(self.fin + 'click_log/Vali_log.json')
            vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair.json')
            if self.topK > 0:
                vali_log = vali_log[vali_log['rankPosition'] < self.topK]
                vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair_topk.json')
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


    def _train_iteration(self, input_train_loader):
        # strat training
        dur = []
        for epoch in range(self.epoch):
            if epoch >= 3:
                t0 = time.time()
            
            loss_log = []
            self.model.train()

            for _id, batch in enumerate(input_train_loader):
                train_loss = self._train_one_batch(batch)
                loss_log.append(train_loss.item())

            # evaluate performance on validation click log
            if self.pairwise:
                val_loss, eval_result = self.vali_tool.evaluate(self.model, 
                                                                loss_type=BPRLoss)
            else:
                pass
            ndcg_val = eval_result['NDCG'][self.topK]
            mrr_val = eval_result['MRR'][self.topK]
            precision_val = eval_result['Precision'][self.topK]

            if self.schedul_mode == 'max':
                self.scheduler.step(ndcg_val)
            elif self.schedul_mode == 'min':
                self.scheduler.step(val_loss)

            if epoch >= self.epoch_start:
                self.early_stopping(ndcg_val*(-1), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break 
            
            if epoch >= 3:
                dur.append(time.time() - t0)
            """
            for name, parms in self.model.named_parameters():
	                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            """
            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
                    "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
                                                    ndcg_val, mrr_val, precision_val))
    
    def _train_one_batch(self, batch):
        self.optim.zero_grad()
        if self.pairwise:
            BPR_lossfunc = BPRLoss(weight=batch[2])
            output_posi = self.model(batch[0],batch[3])
            output_nega = self.model(batch[1],batch[4])
            output_posi = output_posi.view(batch[0].size(0))
            output_nega = output_nega.view(batch[1].size(0))
            train_loss = BPR_lossfunc(output_posi, output_nega)
        else:
            pass
        train_loss.backward()
        self.optim.step()
        return train_loss

    
    def _load_and_wrap_train(self):
        if self.pairwise:
            train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair.json')
            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            if self.topK > 0:
                train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair_topk.json')
                train_log = train_log[train_log['rankPosition'] < self.topK]
            # if self.sparse_tag:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair['pos_feature']),
            #                                         neg_feature=get_sparse_feature(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['click_diff']))
            # else:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair['pos_feature']),
            #                                         neg_feature=torch.Tensor(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['click_diff']))
            input_train = Wrap_Dataset_Pairwise3(train_log_pair['pos_did'].to_list(), 
                                                    train_log_pair['neg_did'].to_list(),
                                                    train_log_pair['pos_pos'].to_list(),
                                                    train_log_pair['neg_pos'].to_list(),
                                                    train_log_pair['rel_diff'].to_list(),
                                                    train_log,
                                                    sparse_tag=self.sparse_tag)
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        else:
            pass
        return input_train_loader

    def _init_train_env(self):
            model = rank_model3(self.in_size, self.hidden_size, self.drop_out, self.topK)
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
                                        verbose=True)
            early_stopping = EarlyStopping(self.fout, self.patience, verbose=True) 
            return model, optim, scheduler, early_stopping

if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = Unify(args)
    learner.train()