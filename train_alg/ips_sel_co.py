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
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2, Wrap_Dataset_Pairwise4
from utils.get_sparse import get_sparse_feature
from utils.early_stop import EarlyStopping
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.loss_func import BPRLoss, HingeLoss,BPRLoss_log_norm, BPRLoss_var_reg, BPRLoss_sel,Dual_BCELoss,Dual_BPRLoss
from models.toy_model import rank_model, rank_model2, rank_model2_2, rank_model_pro, rank_model_pro2

class IPS_sel_co(Learning_Alg):
    def __init__(self, args):
        super(IPS_sel_co, self).__init__(args)
        
    def train(self):
        t0 = time.time()
        self.test_tool, self.in_size = super(IPS_sel_co, self)._load_test_and_get_tool()
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
        super(IPS_sel_co, self)._test_and_save()
    
    def _train_iteration(self, input_train_loader):
        # strat training
        dur = []
        print('STEP I: Training Selection Model...')
        for epoch in range(self.epoch):
            if epoch >= 3:
                t0 = time.time()
            
            loss_log = []
            self.model.train()

            for _id, batch in enumerate(input_train_loader):
                # print('-----')
                train_loss = self._train_one_batch_selection(batch)
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

        print('STEP II: Training Ranking Model...')
        for epoch in range(self.epoch):
            if epoch >= 3:
                t0 = time.time()
            
            loss_log = []
            self.model.train()

            for _id, batch in enumerate(input_train_loader):
                # print('*****')
                train_loss = self._train_one_batch_rank(batch)
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

            


    def _train_one_batch_selection(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            BCE_lossfunc = Dual_BCELoss()
            output_posi_sel, _ = self.model(batch[0])
            output_neg_sel, _ = self.model(batch[1])
            output_posi_sel = output_posi_sel.view(batch[0].size(0))
            output_neg_sel = output_neg_sel.view(batch[1].size(0))
            train_loss =  BCE_lossfunc(output_posi_sel, output_neg_sel, batch[3], batch[4])
        else:
            BCE_lossfunc = BCELoss(weight=batch[2])
            output = self.model(batch[0])
            output = output.view(batch[0].size(0))
            train_loss = BCE_lossfunc(output, batch[1])
        train_loss.backward()
        self.optim.step()
        return train_loss

    def _train_one_batch_rank(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            BPR_lossfunc = Dual_BPRLoss(weight=batch[2])
            _, output_posi = self.model(batch[0])
            _, output_nega = self.model(batch[1])
            output_posi = output_posi.view(batch[0].size(0))
            output_nega = output_nega.view(batch[1].size(0))
            train_loss = BPR_lossfunc(output_posi, output_nega, batch[3], batch[4])
        else:
            BCE_lossfunc = BCELoss(weight=batch[2])
            output = self.model(batch[0])
            output = output.view(batch[0].size(0))
            train_loss = BCE_lossfunc(output, batch[1])
        train_loss.backward()
        self.optim.step()
        return train_loss

    def _init_train_env(self):
        model = rank_model_pro(self.in_size, self.hidden_size, self.drop_out)
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
            train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair.json')
            # train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            train_log = pd.read_json(self.fin + 'json_file/Train.json')
            train_log = format_trans(train_log)
            if self.topK > 0:
                train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair_topk.json')
                train_log_pair_ob = train_log_pair[train_log_pair['tag']==1]
                train_log_pair_unob = train_log_pair.sample(n = int(train_log_pair_ob.shape[0]*self.ratio), replace=False)
                print(train_log_pair_ob.shape)
                print(train_log_pair_unob.shape)
                train_log_pair = pd.concat([train_log_pair_ob, train_log_pair_unob])

                # sample_indicator = train_log_pair['tag'].apply(lambda x: 1 if x==1 else np.random.rand())
                # train_log_pair['sample_indicator'] = sample_indicator
                # train_log_pair = train_log_pair[train_log_pair['sample_indicator']>self.threshold]
                # print(train_log_pair[train_log_pair['sample_indicator']==1].shape)
                # print(train_log_pair[train_log_pair['sample_indicator']<1].shape)

                # train_log_pair = train_log_pair[train_log_pair['click_diff']==1]
                # train_log_pair.loc[train_log_pair['click_diff']==0,'rel_diff']=1
                # train_log_pair.loc[train_log_pair['click_diff']==1,'rel_diff']=1
                # train_log = train_log[train_log['rankPosition'] < self.topK]
            # if self.sparse_tag:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair['pos_feature']),
            #                                         neg_feature=get_sparse_feature(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['rel_diff']))
            # else:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair['pos_feature']),
            #                                         neg_feature=torch.Tensor(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['rel_diff']))
            input_train = Wrap_Dataset_Pairwise4(train_log_pair['pos_did'].to_list(), 
                                                    train_log_pair['neg_did'].to_list(),
                                                    train_log_pair['pos_sel'].to_list(),
                                                    train_log_pair['neg_sel'].to_list(),
                                                    train_log_pair['rel_diff'].to_list(),
                                                    train_log_pair['tag'].to_list(),
                                                    train_log,
                                                    sparse_tag=self.sparse_tag)
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        else:
            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            if self.topK > 0:
                train_log = train_log[train_log['rankPosition'] < self.topK]
            
            input_train = Wrap_Dataset_clickLog(doc_tensor=torch.Tensor(train_log['feature']), 
                                            click_tensor = torch.Tensor(train_log['isClick']), 
                                            ips_tensor = torch.Tensor(train_log['ips_weight']))
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return input_train_loader


    # def _load_and_wrap_train(self):
    #     if self.pairwise:
    #         train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair.json')
    #         train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
    #         if self.topK > 0:
    #             train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair_topk.json')
    #             sample_indicator = train_log_pair['tag'].apply(lambda x: 1 if x==1 else np.random.rand())
    #             train_log_pair['sample_indicator'] = sample_indicator
    #             train_log_pair = train_log_pair[train_log_pair['sample_indicator']>self.threshold]
    #             train_log_observe = train_log_pair[train_log_pair['tag']==1]
    #             train_log_unobserve = train_log_pair[train_log_pair['tag']==0]
    #             print(train_log_observe.shape)
    #             print(train_log_unobserve.shape)
    #             # train_log_pair = train_log_pair[train_log_pair['click_diff']==1]
    #             # train_log_pair.loc[train_log_pair['click_diff']==0,'rel_diff']=1
    #             # train_log_pair.loc[train_log_pair['click_diff']==1,'rel_diff']=1
    #             # train_log = train_log[train_log['rankPosition'] < self.topK]
    #         # if self.sparse_tag:
    #         #     input_train = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair['pos_feature']),
    #         #                                         neg_feature=get_sparse_feature(train_log_pair['neg_feature']),  
    #         #                                         rel_diff=torch.Tensor(train_log_pair['rel_diff']))
    #         # else:
    #         #     input_train = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair['pos_feature']),
    #         #                                         neg_feature=torch.Tensor(train_log_pair['neg_feature']),  
    #         #                                         rel_diff=torch.Tensor(train_log_pair['rel_diff']))
    #         input_train_observe = Wrap_Dataset_Pairwise4(train_log_observe['pos_did'].to_list(), 
    #                                                 train_log_observe['neg_did'].to_list(),
    #                                                 train_log_observe['pos_sel'].to_list(),
    #                                                 train_log_observe['neg_sel'].to_list(),
    #                                                 train_log_observe['rel_diff'].to_list(),
    #                                                 train_log_observe['tag'].to_list(),
    #                                                 train_log,
    #                                                 sparse_tag=self.sparse_tag)
    #         input_train_loader_observe = DataLoader(input_train_observe, 
    #                                         batch_size=self.batch_size, 
    #                                         shuffle=True)

    #         input_train_unobserve = Wrap_Dataset_Pairwise4(train_log_unobserve['pos_did'].to_list(), 
    #                                                 train_log_unobserve['neg_did'].to_list(),
    #                                                 train_log_unobserve['pos_sel'].to_list(),
    #                                                 train_log_unobserve['neg_sel'].to_list(),
    #                                                 train_log_unobserve['rel_diff'].to_list(),
    #                                                 train_log_unobserve['tag'].to_list(),
    #                                                 train_log,
    #                                                 sparse_tag=self.sparse_tag)
    #         input_train_loader_unobserve = DataLoader(input_train_unobserve, 
    #                                         batch_size=self.batch_size, 
    #                                         shuffle=True)
    #     else:
    #         train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
    #         if self.topK > 0:
    #             train_log = train_log[train_log['rankPosition'] < self.topK]
            
    #         input_train = Wrap_Dataset_clickLog(doc_tensor=torch.Tensor(train_log['feature']), 
    #                                         click_tensor = torch.Tensor(train_log['isClick']), 
    #                                         ips_tensor = torch.Tensor(train_log['ips_weight']))
    #         input_train_loader = DataLoader(input_train, 
    #                                         batch_size=self.batch_size, 
    #                                         shuffle=True)
    #     return input_train_loader_observe, input_train_loader_unobserve


if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = IPS_sel_co(args)
    learner.train()