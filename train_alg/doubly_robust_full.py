import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import json
import copy
from torch.nn import BCELoss,BCEWithLogitsLoss,MSELoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
import sys
sys.path.append("..")
import utils.click_model as CM
from models.base_learn_alg import Learning_Alg
from simulate.estimate_rel import RandomizedRelevanceEstimator
from utils.load_data import load_data_forEstimate
from utils.arg_parser import config_param_parser
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise
from utils.early_stop import EarlyStopping
from utils.get_sparse import get_sparse_feature
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.loss_func import BPRLoss, HingeLoss, Unbias_MSELoss,BPRLoss_log_norm
from models.toy_model import rank_model


class Doubly_Robust_Full(Learning_Alg):
    def __init__(self, args):
        super(Doubly_Robust_Full, self).__init__(args)
        self.rel_estimator = args.rel_estimator
        self.pairwise = True

    def train(self):
        self.test_tool, self.in_size = super(Doubly_Robust_Full, self)._load_test_and_get_tool()

        train_log_estimate, vali_log_estimate = self._pre_train_and_estimate()
        self.vali_tool = self._load_vali_and_get_tool(vali_log_estimate)
        input_train_loader_topK, input_train_loader_outK = self._load_and_wrap_train(train_log_estimate)
        self.model, self.optim, self.scheduler, self.early_stopping = super(Doubly_Robust_Full, self)._init_train_env()
        self._train_iteration(input_train_loader_topK, input_train_loader_outK)
        super(Doubly_Robust_Full, self)._test_and_save()

        """
        """

    def _train_iteration(self, input_train_loader_topK, input_train_loader_outK):
        # strat training
        dur = []
        for epoch in range(self.epoch):
            if epoch >= 3:
                t0 = time.time()
            
            loss_log = []
            self.model.train()

            for _id, batch in enumerate(input_train_loader_topK):
                train_loss = self._train_one_batch_topk(batch)
                loss_log.append(train_loss.item())


            for _id, batch in enumerate(input_train_loader_outK):
                train_loss = self._train_one_batch_outk(batch)
                loss_log.append(train_loss.item())



            # evaluate performance on validation click log
            if self.pairwise:
                val_loss, eval_result = self.vali_tool.evaluate(self.model, 
                                                                loss_type=BPRLoss)
            else:
                val_loss, eval_result = self.vali_tool.evaluate(self.model, 
                                                                loss_type=BCELoss)
            ndcg_val = eval_result['NDCG'][10]
            mrr_val = eval_result['MRR'][10]
            precision_val = eval_result['Precision'][10]
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

            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
                    "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
                                                    ndcg_val, mrr_val, precision_val))

    def _train_one_batch_topk(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            BPR_lossfunc = BPRLoss_log_norm(weight=batch[2])
            #BPR_lossfunc = BPRLoss()
            output_posi = self.model(batch[0])
            output_nega = self.model(batch[1])
            output_posi = output_posi.view(batch[0].size(0))
            output_nega = output_nega.view(batch[1].size(0))
            train_loss = BPR_lossfunc(output_posi, output_nega)
        else:
            pass
        train_loss.backward()
        self.optim.step()
        return train_loss


    def _train_one_batch_outk(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            BPR_lossfunc = BPRLoss(weight=batch[2])
            #BPR_lossfunc = BPRLoss()
            output_posi = self.model(batch[0])
            output_nega = self.model(batch[1])
            output_posi = output_posi.view(batch[0].size(0))
            output_nega = output_nega.view(batch[1].size(0))
            train_loss = BPR_lossfunc(output_posi, output_nega)
        else:
            pass
        train_loss.backward()
        self.optim.step()
        return train_loss

    def _load_vali_and_get_tool(self, vali_log_estimate):
        if self.pairwise:
            #vali_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            print('Construct pairwise format for Vali...')
            vali_log_pair = self._get_pair(vali_log_estimate)
            vali_tool = Vali_Evaluator(vali_log_estimate, 
                                       self.eval_positions, 
                                       use_cuda=self.use_cuda, 
                                       with_weight='dr', 
                                       pair_wise=True, pair_df=vali_log_pair)
        else:
            pass
        return vali_tool

    def _load_and_wrap_train(self, train_log_estimate):
        if self.pairwise:
            print('Construct pairwise format for Train...')
            train_log_pair = self._get_pair(train_log_estimate)

            train_log_pair_topk = copy.deepcopy(train_log_pair[train_log_pair['tag']==1])
            print(train_log_pair_topk.shape)
            if self.sparse_tag:
                input_train_topk = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair_topk['pos_feature'].to_list()),
                                                neg_feature=get_sparse_feature(train_log_pair_topk['neg_feature'].to_list()),  
                                                rel_diff=torch.Tensor((train_log_pair_topk['rel_diff'] * train_log_pair_topk['dr_weight']).to_list()))
            else:
                input_train_topk = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair_topk['pos_feature'].to_list()),
                                                    neg_feature=torch.Tensor(train_log_pair_topk['neg_feature'].to_list()),  
                                                    rel_diff=torch.Tensor((train_log_pair_topk['rel_diff'] * train_log_pair_topk['dr_weight']).to_list()))
            input_train_loader_topK = DataLoader(input_train_topk, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)

            train_log_pair_outk = copy.deepcopy(train_log_pair[train_log_pair['tag']==0])
            print(train_log_pair_outk.shape)
            if self.sparse_tag:
                input_train_outk = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair_outk['pos_feature'].to_list()),
                                                    neg_feature=get_sparse_feature(train_log_pair_outk['neg_feature'].to_list()),  
                                                    rel_diff=torch.Tensor((train_log_pair_outk['rel_diff'] * train_log_pair_outk['dr_weight']).to_list()))
            else:
                input_train_outk = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair_outk['pos_feature'].to_list()),
                                                    neg_feature=torch.Tensor(train_log_pair_outk['neg_feature'].to_list()),  
                                                    rel_diff=torch.Tensor((train_log_pair_outk['rel_diff'] * train_log_pair_outk['dr_weight']).to_list()))
            input_train_loader_outK = DataLoader(input_train_outk, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        else:
            pass 
        return input_train_loader_topK, input_train_loader_outK

    def _get_pair(self, df):
        df_pair = pd.DataFrame([],columns=['sid','qid','tag','dr_weight', 'rel_diff', 'pos_did', 'pos_feature', 'neg_did', 'neg_feature'])

        df_sort_topk = df[df['rankPosition']<self.topK].groupby('sid').apply(lambda x: x.sort_values(by = ['dr_score'], ascending=[False]))
        df_sort_topk.reset_index(drop=True, inplace=True)    
        df_sort_outk = df[df['rankPosition']>=self.topK].groupby('sid').apply(lambda x: x.sort_values(by = ['dr_score'], ascending=[False]))
        df_sort_outk.reset_index(drop=True, inplace=True) 
        rows_pair = []

        # processing Top-K samples
        for _, row_out in tqdm(df_sort_topk.iterrows(), total=df_sort_topk.shape[0]):
            if row_out['isClick'] > 0:
                for _, row_in in df_sort_topk[df_sort_topk['sid']==row_out['sid']].iterrows():
                    if row_out['dr_score'] > row_in['dr_score']:
                        rows_pair.append(self._assign_val(row_out, row_in, True))
            else:
                continue


        # processing out-K samples
        all_sid = df_sort_outk['sid'].unique()
        for sid in tqdm(all_sid, total=len(all_sid)):
            cur_part = df_sort_outk[df_sort_outk['sid']==sid]
            length = cur_part.shape[0]
            start_idx = cur_part.iloc[0].name
            for idx, row_out in cur_part.iterrows():
                cur_idx = idx
                tag_list = [False] * (length - (cur_idx - start_idx + 1))
                rows_cur = [row_out] * (length - (cur_idx - start_idx + 1))
                rows_left = [r for _,r in cur_part.iloc[cur_idx - start_idx + 1:length].iterrows()]
                rows_pair.extend(list(map(self._assign_val, rows_cur, rows_left, tag_list)))


        # assign data into DataFrame
        rows_pair = np.array(rows_pair, dtype=object)
        for i, col_name in enumerate(df_pair.columns):
            df_pair[col_name] = rows_pair[:,i]
        df_pair = df_pair[df_pair['rel_diff']>=self.threshold]
        df_pair = df_pair.sort_values(by=['sid'], ascending=[True])
        df_pair.reset_index(drop=True, inplace=True) 

        return df_pair

    @staticmethod
    def _assign_val(row_out, row_in, isTopK=True):
        if isTopK==True:
            tag = 1
        else:
            tag = 0
        row_new = [ row_out['sid'], row_out['qid'], tag, row_out['dr_weight'],
                    row_out['dr_score']*row_out['dr_label'] - row_in['dr_score']*row_out['dr_label'], 
                    row_out['did'], row_out['feature'], 
                    row_in['did'], row_in['feature']]
        return row_new

    def _random_estimate(self):
        with open('pbm.json') as f:	
            clickmodel_dict = json.load(f)
        pbm = CM.loadModelFromJson(clickmodel_dict)
        train_label_data = load_data_forEstimate(self.fin + 'json_file/Train.json')
        vali_label_data = load_data_forEstimate(self.fin + 'json_file/Vali.json')

        print('Estimating Relevance for Train...')
        train_session_num = self.session_num
        train_estimator = RandomizedRelevanceEstimator()
        train_estimator.estimateRelevanceFromModel(pbm, train_label_data, train_session_num)
        train_estimate_data = train_estimator.outputResultToData(train_label_data)
        df_train = pd.DataFrame(train_estimate_data)
        df_train.rename(columns={'queryID':'qid', 'docID':'did'}, inplace = True)
        #df_train['estimate_label'] = df_train['estimate_label'].apply(lambda x: 1 if x > 0 else 0)
        df_train = df_train[['qid','did','label','estimate_label','rankPosition','feature']]

        print('Estimating Relevance for Vali...')
        vali_session_num = train_session_num * (vali_label_data.data_size / train_label_data.data_size)
        vali_estimator = RandomizedRelevanceEstimator()
        vali_estimator.estimateRelevanceFromModel(pbm, vali_label_data, int(vali_session_num))
        vali_estimate_data = vali_estimator.outputResultToData(vali_label_data)
        df_vali = pd.DataFrame(vali_estimate_data)
        df_vali.rename(columns={'queryID':'qid', 'docID':'did'}, inplace = True)
        #df_vali['estimate_label'] = df_vali['estimate_label'].apply(lambda x: 1 if x > 0 else 0)
        df_vali = df_vali[['qid','did','label','estimate_label','rankPosition','feature']]

        df_train.to_json(self.fin + 'click_log/train_log_rand_estimate.json')
        df_vali.to_json(self.fin + 'click_log/vali_log_rand_estimate.json')
        
        return df_train, df_vali


    def _pre_train_and_estimate(self):

        if self.rel_estimator=='random':
            """
            train an Pseudo full information model to predict the relevance for all data 
            """
            print('Pretrain an Pseudo Full Information Model ...')
            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            vali_log = pd.read_json(self.fin + 'click_log/Vali_log.json')
            df_train, df_vali = self._random_estimate()
            ips_vali_tool = Vali_Fullinfo_Evaluator(df_vali, 
                                                self.eval_positions, 
                                                use_cuda=self.use_cuda)
            if self.sparse_tag:
                input_train = Wrap_Dataset_fullInfo(doc_tensor=get_sparse_feature(df_train['feature']), 
                                                    label_tensor = torch.Tensor(df_train['estimate_label']))
            else:
                input_train = Wrap_Dataset_fullInfo(doc_tensor=torch.Tensor(df_train['feature']), 
                                                    label_tensor = torch.Tensor(df_train['estimate_label']))
            ips_input_train_loader = DataLoader(input_train, 
                                            batch_size=self.pre_bs, 
                                            shuffle=True)
                
            ips_model = rank_model(self.in_size, self.hidden_size, self.drop_out)
            if self.use_cuda:
                ips_model = ips_model.cuda()
            if self.optimizer == 'adam':
                ips_optim = Adam(ips_model.parameters(), lr=self.pre_lr, weight_decay=self.pre_wd)
            elif self.optimizer == 'sgd':
                ips_optim = SGD(ips_model.parameters(), lr=self.pre_lr, weight_decay=self.pre_wd)
            ips_scheduler = ReduceLROnPlateau(ips_optim, 
                                        patience=10, 
                                        mode=self.schedul_mode,
                                        verbose=True)
            ips_early_stopping = EarlyStopping(self.patience, verbose=True) 
            dur = []
            for epoch in range(self.epoch):
                if epoch >= 3:
                    t0 = time.time()
                
                loss_log = []
                ips_model.train()
                for _id, batch in enumerate(ips_input_train_loader):
                    ips_optim.zero_grad()
                    #BCE_lossfunc = BCELoss()
                    MSE_lossfunc = Unbias_MSELoss()
                    output = ips_model(batch[0])
                    output = output.view(batch[0].size(0))
                    #train_loss = BCE_lossfunc(output, batch[1])
                    train_loss = MSE_lossfunc(output, batch[1])
                    train_loss.backward()
                    ips_optim.step()
                    loss_log.append(train_loss.item())

                # evaluate performance on validation click log
                val_loss, eval_result = ips_vali_tool.evaluate(ips_model, 
                                                                loss_type=Unbias_MSELoss)
                ndcg_val = eval_result['NDCG'][10]
                mrr_val = eval_result['MRR'][10]
                precision_val = eval_result['Precision'][10]
                if self.schedul_mode == 'max':
                    ips_scheduler.step(ndcg_val)
                elif self.schedul_mode == 'min':
                    ips_scheduler.step(val_loss)

                if epoch >= self.epoch_start:
                    ips_early_stopping(ndcg_val*(-1), ips_model)

                if ips_early_stopping.early_stop:
                    print("Early stopping")
                    break 
                
                if epoch >= 3:
                    dur.append(time.time() - t0)

                print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
                        "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
                                                        ndcg_val, mrr_val, precision_val))     
            # predict estimate label for train and vali
            ips_model.load_state_dict(torch.load('checkpoint.pt'))
            ips_model.eval()
            if self.use_cuda:
                train_predict = ips_model(torch.Tensor(train_log['feature']).cuda())
                vali_predict = ips_model(torch.Tensor(vali_log['feature']).cuda())
            else:
                train_predict = ips_model(torch.Tensor(train_log['feature']).cpu())
                vali_predict = ips_model(torch.Tensor(vali_log['feature']).cpu())                
            train_predict = train_predict.cpu()
            train_predict = train_predict.view(-1).tolist()
            vali_predict = vali_predict.cpu()
            vali_predict = vali_predict.view(-1).tolist()
            train_log['estimate_score'] = train_predict
            vali_log['estimate_score'] = vali_predict
            dr_score_train = []
            lamda_train = []
            label_train = []
            for _, row in train_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']*(row['isClick'] - row['isObserve']*row['estimate_score']) + row['estimate_score']
                    lamda = 1
                    p_label = 1
                else:
                    wt = row['estimate_score']
                    lamda = self.lamda
                    p_label = 1
                #wt = row['ips_weight'] * (row['isClick'] - row['estimate_label']) + row['estimate_label']
                label_train.append(p_label)
                dr_score_train.append(wt)
                lamda_train.append(lamda)
            dr_score_vali = []
            lamda_vali = []
            label_vali = []
            for _, row in vali_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']*(row['isClick'] - row['isObserve']*row['estimate_score']) + row['estimate_score']
                    lamda = 1 
                    p_label = 1
                else:
                    wt = row['estimate_score']
                    lamda = self.lamda
                    p_label = 1
                #wt = row['ips_weight'] * (row['isClick'] - row['estimate_label']) + row['estimate_label']
                label_vali.append(p_label)
                dr_score_vali.append(wt)
                lamda_vali.append(lamda)
            
            train_log['dr_label'] = label_train
            train_log['dr_score'] = dr_score_train
            train_log['dr_weight'] = lamda_train

            vali_log['dr_label'] = label_vali
            vali_log['dr_score'] = dr_score_vali 
            vali_log['dr_weight'] = lamda_vali       
            
            train_log.to_json('../datasets/result2/train_log_rand_estimate.json')
            vali_log.to_json('../datasets/result2/vali_log_rand_estimate.json')
            torch.save(ips_model.state_dict(), '{}_pretrainFullinfo.pt'.format(self.fout))      

        elif self.rel_estimator=='ips':
            """
            train an ips model to predict the relevance for all data
            """
            print('Pretrain an IPS Model ...')
            vali_log = pd.read_json(self.fin + 'click_log/Vali_log.json')
            vali_log_topK = vali_log[vali_log['rankPosition'] < self.topK]
            #vali_log_out_topK = vali_log[vali_log['rankPosition'] >= self.topK]
            ips_vali_tool = Vali_Evaluator(vali_log_topK, 
                                        self.eval_positions, 
                                        use_cuda=self.use_cuda, 
                                        with_weight='ips')

            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            train_log_topK = train_log[train_log['rankPosition'] < self.topK]
            #train_log_out_topK = train_log[train_log['rankPosition'] >= self.topK]
            if self.sparse_tag:
                input_train = Wrap_Dataset_clickLog(doc_tensor=get_sparse_feature(train_log_topK['feature']), 
                                                click_tensor = torch.Tensor(train_log_topK['isClick']), 
                                                ips_tensor = torch.Tensor(train_log_topK['ips_weight_for_train']))
            else:
                input_train = Wrap_Dataset_clickLog(doc_tensor=torch.Tensor(train_log_topK['feature']), 
                                                click_tensor = torch.Tensor(train_log_topK['isClick']), 
                                                ips_tensor = torch.Tensor(train_log_topK['ips_weight_for_train']))
            ips_input_train_loader = DataLoader(input_train, 
                                            batch_size=self.pre_bs, 
                                            shuffle=True)
            ips_model = rank_model(self.in_size, self.hidden_size, self.drop_out)
            if self.use_cuda:
                ips_model = ips_model.cuda()
            if self.optimizer == 'adam':
                ips_optim = Adam(ips_model.parameters(), lr=self.pre_lr, weight_decay=self.pre_wd)
            elif self.optimizer == 'sgd':
                ips_optim = SGD(ips_model.parameters(), lr=self.pre_lr, weight_decay=self.pre_wd)
            ips_scheduler = ReduceLROnPlateau(ips_optim, 
                                            patience=10, 
                                            mode=self.schedul_mode,
                                            verbose=True)
            # ips_early_stopping = EarlyStopping(self.patience, verbose=True) 
            ips_early_stopping = EarlyStopping(10, verbose=True)

            dur = []
            for epoch in range(self.epoch):
                if epoch >= 3:
                    t0 = time.time()
                
                loss_log = []
                ips_model.train()
                for _id, batch in enumerate(ips_input_train_loader):
                    ips_optim.zero_grad()
                    #BCE_lossfunc = BCELoss(weight=batch[2])
                    MSE_lossfunc = Unbias_MSELoss()
                    output = ips_model(batch[0])
                    output = output.view(batch[0].size(0))
                    train_loss = MSE_lossfunc(output, batch[1]*batch[2])
                    train_loss.backward()
                    ips_optim.step()
                    loss_log.append(train_loss.item())

                # evaluate performance on validation click log
                val_loss, eval_result = ips_vali_tool.evaluate(ips_model, 
                                                                loss_type=Unbias_MSELoss)
                ndcg_val = eval_result['NDCG'][10]
                mrr_val = eval_result['MRR'][10]
                precision_val = eval_result['Precision'][10]
                if self.schedul_mode == 'max':
                    ips_scheduler.step(ndcg_val)
                elif self.schedul_mode == 'min':
                    ips_scheduler.step(val_loss)

                if epoch >= self.epoch_start:
                    ips_early_stopping(ndcg_val*(-1), ips_model)

                if ips_early_stopping.early_stop:
                    print("Early stopping")
                    break 
                
                if epoch >= 3:
                    dur.append(time.time() - t0)

                print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
                        "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
                                                        ndcg_val, mrr_val, precision_val))     
            
            # predict estimate label for train and vali
            ips_model.load_state_dict(torch.load('checkpoint.pt'))
            ips_model.eval()
            if self.use_cuda:
                train_predict = ips_model(torch.Tensor(train_log['feature']).cuda())
                vali_predict = ips_model(torch.Tensor(vali_log['feature']).cuda())
            else:
                train_predict = ips_model(torch.Tensor(train_log['feature']).cpu())
                vali_predict = ips_model(torch.Tensor(vali_log['feature']).cpu())               
            train_predict = train_predict.cpu()
            train_predict = train_predict.view(-1).tolist()
            vali_predict = vali_predict.cpu()
            vali_predict = vali_predict.view(-1).tolist()
            train_log['estimate_score'] = train_predict
            vali_log['estimate_score'] = vali_predict
            dr_score_train = []
            lamda_train = []
            label_train = []
            for _, row in train_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']*(row['isClick'] - row['isObserve']*row['estimate_score']) + row['estimate_score']
                    lamda = 1
                    p_label = 1
                else:
                    wt = row['estimate_score']
                    lamda = self.lamda
                    p_label = 1
                #wt = row['ips_weight'] * (row['isClick'] - row['estimate_label']) + row['estimate_label']
                label_train.append(p_label)
                dr_score_train.append(wt)
                lamda_train.append(lamda)
            dr_score_vali = []
            lamda_vali = []
            label_vali = []
            for _, row in vali_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']*(row['isClick'] - row['isObserve']*row['estimate_score']) + row['estimate_score']
                    lamda = 1 
                    p_label = 1
                else:
                    wt = row['estimate_score']
                    lamda = self.lamda
                    p_label = 1
                #wt = row['ips_weight'] * (row['isClick'] - row['estimate_label']) + row['estimate_label']
                label_vali.append(p_label)
                dr_score_vali.append(wt)
                lamda_vali.append(lamda)
            
            train_log['dr_label'] = label_train
            train_log['dr_score'] = dr_score_train
            train_log['dr_weight'] = lamda_train

            vali_log['dr_label'] = label_vali
            vali_log['dr_score'] = dr_score_vali 
            vali_log['dr_weight'] = lamda_vali      
            
            train_log.to_json('../datasets/result2/train_log_ips_estimate.json')
            vali_log.to_json('../datasets/result2/vali_log_ips_estimate.json')
            torch.save(ips_model.state_dict(), '{}_pretrainIPS.pt'.format(self.fout))

        return train_log, vali_log


if __name__=='__main__':
    parser = config_param_parser()
    args = parser.parse_args()
    dr = Doubly_Robust_Full(args)
    dr.train()

    