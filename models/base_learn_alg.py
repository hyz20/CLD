import sys
import os
import random
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import Module
from torch.nn import init
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCELoss,BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from utils.set_seed import setup_seed
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2
from utils.early_stop import EarlyStopping
from utils.arg_parser import config_param_parser
from utils.pairwise_trans import get_pair, get_pair_fullinfo
from utils.trans_format import format_trans
from utils.get_sparse import get_sparse_feature
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.toy_model import rank_model,rank_model2,rank_model_pro,rank_model_linear
from models.loss_func import BPRLoss, HingeLoss


class Learning_Alg(object):
    
    def __init__(self, args):
        self.fin = args.fin
        self.fout = args.fout
        self.eval_positions = args.eval_positions
        self.use_cuda = args.use_cuda
        self.pairwise = args.pairwise # add this arg
        self.topK = args.topK
        self.hidden_size = args.hidden_size
        self.drop_out = args.drop_out
        self.optimizer = args.optimizer
        self.schedul_mode = args.schedul_mode
        self.lamda = args.lamda
        self.lr = args.lr
        self.pre_lr = args.pre_lr
        self.weight_decay = args.weight_decay
        self.pre_wd = args.pre_wd
        self.patience = args.patience
        self.epoch = args.epoch
        self.epoch_start = args.epoch_start
        self.batch_size = args.batch_size
        self.pre_bs = args.pre_bs
        self.train_alg = args.train_alg # ??
        self.sparse_tag = args.sparse_tag
        self.threshold = args.threshold
        self.session_num = args.session_num
        self.C = args.C
        self.rho = args.rho
        self.continue_tag = args.continue_tag
        self.train_method = args.train_method
        self.ratio = args.ratio
        self.randseed = args.randseed
        
    def train(self):
        self.test_tool, self.in_size = self._load_test_and_get_tool()
        # self.vali_tool = self._load_vali_and_get_tool()
        input_train_loader = self._load_and_wrap_train()
        # input_train = self._load_and_wrap_train()

        self.model, self.optim, self.scheduler, self.early_stopping = self._init_train_env()
        self._train_iteration(input_train_loader)
        self._test_and_save()
    
    def _test_and_save(self):
        # load best model
        self.model.load_state_dict(torch.load('{}_checkpoint.pt'.format(self.fout)))

        # evaluate model on test set
        test_result = self.test_tool.evaluate(self.model)

        # save model and result as given path
        #print('test_loss:', test_loss)
        torch.save(self.model.state_dict(), '{}_model.pt'.format(self.fout))
        test_result.to_json('{}_result.json'.format(self.fout), indent=4)

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
            
            if epoch >= 3:
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
    
    def _train_one_batch(self, batch):
        self.optim.zero_grad()
        if self.pairwise:
            BPR_lossfunc = BPRLoss(weight=batch[2])
            output_posi = self.model(batch[0])
            output_nega = self.model(batch[1])
            output_posi = output_posi.view(batch[0].size(0))
            output_nega = output_nega.view(batch[1].size(0))
            train_loss = BPR_lossfunc(output_posi, output_nega)
        else:
            BCE_lossfunc = BCELoss()
            output = self.model(batch[0])
            output = output.view(batch[0].size(0))
            train_loss = BCE_lossfunc(output, batch[1])
        train_loss.backward()
        self.optim.step()
        return train_loss
    
    def _init_train_env(self):
        model = rank_model(self.in_size, self.hidden_size, self.drop_out)
        # model = rank_model_linear(self.in_size, self.hidden_size, self.drop_out)
        # model = rank_model_pro(self.in_size, self.hidden_size, self.drop_out)
        # model = rank_model2(self.in_size, self.hidden_size, self.drop_out)
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
        
    def _load_test_and_get_tool(self):
        test_dat = pd.read_json(self.fin + 'json_file/Test.json')
        test_dat = format_trans(test_dat,mode='test')
        test_tool = Test_Evaluator(test_dat, 
                                   self.eval_positions, 
                                   use_cuda=self.use_cuda)
        in_size = len(test_dat['feature'][0])
        # if isinstance(test_dat['feature'][0][0], list):
        #     # if load feature is sparse format
        #     in_size = 699
        # else:
        #     in_size = len(test_dat['feature'][0])
        return test_tool, in_size
    
    def _load_vali_and_get_tool(self):
        if self.pairwise:
            vali_dat = pd.read_json(self.fin + 'json_file/Vali.json')
            vali_dat = format_trans(vali_dat)
            vali_dat_pair = pd.read_json(self.fin + 'json_file/Vali_pair.json')
            vali_tool = Vali_Fullinfo_Evaluator(vali_dat, 
                                                self.eval_positions, 
                                                use_cuda=self.use_cuda, 
                                                pair_wise=True, pair_df=vali_dat_pair)
        else:
            vali_dat = pd.read_json(self.fin + 'json_file/Vali.json')
            vali_dat = format_trans(vali_dat)
            vali_tool = Vali_Fullinfo_Evaluator(vali_dat, 
                                                self.eval_positions, 
                                                use_cuda=self.use_cuda)
        return vali_tool
        
    def _load_and_wrap_train(self):
        if self.pairwise:
            train_dat_pair = pd.read_json(self.fin + 'json_file/Train_pair.json')
            train_dat_pair = train_dat_pair.sample(frac=0.1, replace=False)
            train_dat = pd.read_json(self.fin + 'json_file/Train.json')
            train_dat = format_trans(train_dat)
            # print(train_dat_pair['pos_feature'].values.shape)
            # print(len(train_dat_pair['pos_feature'][0]))
            # if self.sparse_tag:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_dat_pair['pos_feature']),
            #                                         neg_feature=get_sparse_feature(train_dat_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_dat_pair['rel_diff']))
            # else:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_dat_pair['pos_feature']),
            #                                         neg_feature=torch.Tensor(train_dat_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_dat_pair['rel_diff']))
            input_train = Wrap_Dataset_Pairwise2(train_dat_pair['pos_did'].to_list(), 
                                                    train_dat_pair['neg_did'].to_list(),
                                                    train_dat_pair['rel_diff'].to_list(),
                                                    train_dat,
                                                    sparse_tag=self.sparse_tag)
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        else:
            train_dat = pd.read_json(self.fin + 'json_file/Train.json')
            train_dat = format_trans(train_dat)
            if self.sparse_tag:
                input_train = Wrap_Dataset_fullInfo(doc_tensor=get_sparse_feature(train_dat['feature']), 
                                                    label_tensor = torch.Tensor(train_dat['label']))
            else:
                input_train = Wrap_Dataset_fullInfo(doc_tensor=torch.Tensor(train_dat['feature']), 
                                                    label_tensor = torch.Tensor(train_dat['label']))

            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return input_train_loader


if __name__=="__main__":
    pass