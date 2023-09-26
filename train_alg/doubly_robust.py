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
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2
from utils.early_stop import EarlyStopping
from utils.get_sparse import get_sparse_feature
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.loss_func import BPRLoss, HingeLoss, Unbias_MSELoss,BPRLoss_log_norm, Unbias_MSELoss_reg, BPRLoss_var_reg
from models.toy_model import rank_model, rank_model2, rank_model2_2,rank_model_pro


class Doubly_Robust(Learning_Alg):
    def __init__(self, args):
        super(Doubly_Robust, self).__init__(args)
        self.rel_estimator = args.rel_estimator
        self.pairwise = True

    def train(self):
        self.test_tool, self.in_size = super(Doubly_Robust, self)._load_test_and_get_tool()

        train_log_estimate, vali_log_estimate = self._pre_train_and_estimate()
        self.vali_tool = self._load_vali_and_get_tool(vali_log_estimate)
        input_train_loader_topK, input_train_loader_outK = self._load_and_wrap_train(train_log_estimate)
        self.model, self.optim, self.scheduler, self.early_stopping = super(Doubly_Robust, self)._init_train_env()
        self._train_iteration(input_train_loader_topK, input_train_loader_outK)
        super(Doubly_Robust, self)._test_and_save()

        """
        """

    def _train_iteration(self, input_train_loader_topK, input_train_loader_outK):
        # strat training
        dur = []
        print('start training ...')
        for epoch in range(self.epoch):
            
            if epoch >= 3:
                t0 = time.time()
            
            loss_log = []
            self.model.train()

            for _id, batch in enumerate(input_train_loader_topK):
                train_loss = self._train_one_batch_topk(batch)
                loss_log.append(train_loss.item())

            # if input_train_loader_outK is not None:
            #     for _id, batch in enumerate(input_train_loader_outK):
            #         train_loss = self._train_one_batch_outk(batch)
            #         loss_log.append(train_loss.item())



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

            # print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
            #         "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
            #                                         ndcg_val, mrr_val, precision_val))

            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Test_NDCG@10 {:.4f} | "
                    "Test_MRR@10 {:.4f}| Test_Precision@10 {:.4f} | Test_MAP@10 {:.4f} | Test_NDCG_full {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),ndcg_tst,
                                                    mrr_tst, precision_tst, map_tst, ndcg_full_tst))

    def _train_one_batch_topk(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            #BPR_lossfunc = BPRLoss_log_norm(weight=batch[2])
            #BPR_lossfunc = BPRLoss()
            BPR_lossfunc = BPRLoss(weight=batch[2])
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
            #BPR_lossfunc = BPRLoss_log_norm(weight=batch[2])
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
            vali_log_pair = self._get_pair(vali_log_estimate, mode='vali')
            print(vali_log_pair[vali_log_pair['tag']==1].shape)
            print(vali_log_pair[vali_log_pair['tag']==0].shape)

            vali_tool = Vali_Evaluator(vali_log_estimate[vali_log_estimate['rankPosition']<self.topK],
                                       self.eval_positions, 
                                       use_cuda=self.use_cuda, 
                                       with_weight=self.train_alg, 
                                       pair_wise=True, pair_df=vali_log_pair)
        else:
            pass
        return vali_tool

    def _load_and_wrap_train(self, train_log_estimate):
        if self.pairwise:
            print('Construct pairwise format for Train...')
            train_log_pair = self._get_pair(train_log_estimate, mode='train')

            print(train_log_pair[train_log_pair['tag']==1].shape)
            print(train_log_pair[train_log_pair['tag']==0].shape)
            input_train_topk = Wrap_Dataset_Pairwise2(train_log_pair['pos_did'].to_list(),
                                                        train_log_pair['neg_did'].to_list(), 
                                                        train_log_pair['rel_diff'].to_list(), 
                                                        train_log_estimate,
                                                        sparse_tag=self.sparse_tag)
            input_train_loader_topK = DataLoader(input_train_topk, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
            input_train_loader_outK = input_train_loader_topK
            # train_log_pair_topk = copy.deepcopy(train_log_pair[train_log_pair['tag']==1])
            # print(train_log_pair_topk.shape)
            # input_train_topk = Wrap_Dataset_Pairwise2(train_log_pair_topk['pos_did'].to_list(),
            #                                             train_log_pair_topk['neg_did'].to_list(), 
            #                                             train_log_pair_topk['rel_diff'].to_list(), 
            #                                             train_log_estimate,
            #                                             sparse_tag=self.sparse_tag)
            # input_train_loader_topK = DataLoader(input_train_topk, 
            #                                 batch_size=self.batch_size, 
            #                                 shuffle=True)

            # train_log_pair_outk = copy.deepcopy(train_log_pair[train_log_pair['tag']==0])
            # print(train_log_pair_outk.shape)
            # input_train_outk = Wrap_Dataset_Pairwise2(train_log_pair_outk['pos_did'].to_list(),
            #                                             train_log_pair_outk['neg_did'].to_list(), 
            #                                             train_log_pair_outk['rel_diff'].to_list(), 
            #                                             train_log_estimate,
            #                                             sparse_tag=self.sparse_tag)
            # if train_log_pair_outk.shape[0]>self.batch_size:
            #     input_train_loader_outK = DataLoader(input_train_outk, 
            #                                     batch_size=self.batch_size, 
            #                                     shuffle=True)
            # else:
            #     input_train_loader_outK = None
        else:
            pass 
        return input_train_loader_topK, input_train_loader_outK

    def _get_pair(self, df, mode='train'):
        if mode=='train':
            df_pair = pd.DataFrame([],columns=['sid', 'tag', 'rel_diff', 'pos_did', 'neg_did'])
        elif mode=='vali':
            df_pair = pd.DataFrame([],columns=['sid','tag', 'rel_diff', 'pos_did', 'pos_feature', 'neg_did', 'neg_feature'])


        df_sort_topk = df[df['rankPosition']<self.topK].groupby('sid').apply(lambda x: x.sort_values(by = ['dr_label'], ascending=[False]))
        df_sort_topk.reset_index(drop=True, inplace=True)    
        df_sort_outk = df[df['rankPosition']>=self.topK].groupby('sid').apply(lambda x: x.sort_values(by = ['dr_label'], ascending=[False]))
        df_sort_outk.reset_index(drop=True, inplace=True) 
        rows_pair = []
        #print(df_sort_topk.shape)
        #print(df_sort_outk.shape)


        # processing Top-K samples
        # df_sort_topk['normalized_var'] = df_sort_topk.groupby('sid')['estimate_var'].apply(lambda x: (x-max(x))/(min(x)-max(x)))

        for _, row_out in tqdm(df_sort_topk.iterrows(), total=df_sort_topk.shape[0]):
            if row_out['isClick'] > 0:
                for _, row_in in df_sort_topk[df_sort_topk['sid']==row_out['sid']].iterrows():
                    # if row_in['isClick']==0:
                    if row_out['dr_score']*row_out['dr_label'] > row_in['dr_score']*row_in['dr_label'] :
                        # if mode=='train':
                        #     print(1)
                        rows_pair.append(self._assign_val(row_out, row_in, True, mode=mode))
                        # rows_pair.append(self._assign_val(row_out, row_in, True, mode=mode))
            else:
                continue
        

        # processing out-K samples
        # all_sid = df_sort_outk['sid'].unique()
        # for sid in tqdm(all_sid, total=len(all_sid)):
        #     cur_part = df_sort_outk[df_sort_outk['sid']==sid]
        #     length = cur_part.shape[0]
        #     start_idx = cur_part.iloc[0].name
        #     if (cur_part.head(1)['dr_score'].values - cur_part.tail(1)['dr_score'].values)[0]>=self.threshold * self.lamda:
        #         for idx, row_out in cur_part.iterrows():
        #             cur_idx = idx
        #             rows_left = list(filter(lambda x: x is not None, [r if (row_out['dr_score']*row_out['dr_label'] - r['dr_score']*r['dr_label']) >= self.threshold * self.lamda else None 
        #                                         for _,r in cur_part.iloc[cur_idx - start_idx + 1:length].iterrows()]))
        #             tag_list = [False] * len(rows_left)
        #             rows_cur = [row_out] * len(rows_left)
        #             mode_list = [mode] * len(rows_left)

        #             rows_pair.extend(list(map(self._assign_val, rows_cur, rows_left, tag_list, mode_list)))
        #     else:
        #         continue


        # all_sid = df_sort_outk['sid'].unique()
        # up_quantiles = df_sort_outk.groupby('sid')['dr_score'].quantile(0.98)
        # low_quantiles = df_sort_outk.groupby('sid')['dr_score'].quantile(0.02)
        # threshold_quantiles = (up_quantiles - low_quantiles)
        # #threshold_quantiles = 100
        # #print(threshold_quantiles)

        # for sid in tqdm(all_sid, total=len(all_sid)):
        #     cur_part = df_sort_outk[df_sort_outk['sid']==sid]
        #     length = cur_part.shape[0]
        #     start_idx = cur_part.iloc[0].name
        #     if (cur_part.head(1)['dr_score'].values - cur_part.tail(1)['dr_score'].values)[0] >= threshold_quantiles[sid]:
        #         for idx, row_out in cur_part.iterrows():
        #             cur_idx = idx
        #             rows_left = list(filter(lambda x: x is not None, [r if (row_out['dr_score']*row_out['dr_label'] - r['dr_score']*r['dr_label']) >= threshold_quantiles[sid] else None 
        #                                         for _,r in cur_part.iloc[cur_idx - start_idx + 1:length].iterrows()]))
        #             tag_list = [False] * len(rows_left)
        #             rows_cur = [row_out] * len(rows_left)
        #             mode_list = [mode] * len(rows_left)

        #             rows_pair.extend(list(map(self._assign_val, rows_cur, rows_left, tag_list, mode_list)))
        #     else:
        #         continue
        if mode=='train':
            # df_sort_outk['normalized_var'] = df_sort_outk.groupby('sid')['estimate_var'].apply(lambda x: (x-max(x))/(min(x)-max(x)))
            df_sort_outk = df_sort_outk.groupby('sid').apply(lambda x: x.sort_values(by = ['dr_label'], ascending=[False]))
            df_sort_outk.reset_index(drop=True, inplace=True)
            # df_sort_outk = df_sort_outk[df_sort_outk['estimate_var']<=0.0005]

            up_quantiles = df_sort_outk.groupby('sid')['dr_label'].quantile(0.95)
            low_quantiles = df_sort_outk.groupby('sid')['dr_label'].quantile(0.05)
            threshold_quantiles = (up_quantiles - low_quantiles)
            all_sid = df_sort_outk['sid'].unique()
            # threshold = 0

            df_sort_outk_selected = df_sort_outk
            # df_sort_outk_selected = df_sort_outk.groupby('sid').apply(lambda x: x.sort_values('estimate_var', ascending=True).head(10))
            # df_sort_outk_selected.reset_index(drop=True, inplace=True)
            # df_sort_outk_selected['normalized_var'] = df_sort_outk_selected.groupby('sid')['estimate_var'].apply(lambda x: (x-max(x))/(min(x)-max(x)))
            # df_sort_outk_selected = df_sort_outk_selected.groupby('sid').apply(lambda x: x.sort_values(by = ['dr_score'], ascending=[False]))
            # df_sort_outk_selected.reset_index(drop=True, inplace=True) 
            
            # for sid in tqdm(all_sid, total=len(all_sid)):
            #     cur_part = df_sort_outk_selected[df_sort_outk_selected['sid']==sid]
            #     length = cur_part.shape[0]
            #     start_idx = cur_part.iloc[0].name
            #     if (cur_part.head(1)['dr_label'].values - cur_part.tail(1)['dr_label'].values)[0] >= threshold_quantiles[sid]:
            #         for idx, row_out in cur_part.iterrows():
            #             cur_idx = idx
            #             rows_left = list(filter(lambda x: x is not None, [r if (row_out['dr_score']*row_out['dr_label'] - r['dr_score']*r['dr_label']) >= threshold_quantiles[sid] else None 
            #                                         for _,r in cur_part.iloc[cur_idx - start_idx + 1:length].iterrows()]))
            #             tag_list = [False] * len(rows_left)
            #             rows_cur = [row_out] * len(rows_left)
            #             mode_list = [mode] * len(rows_left)

            #             rows_pair.extend(list(map(self._assign_val, rows_cur, rows_left, tag_list, mode_list)))
            #     else:
            #         continue

            for sid in tqdm(all_sid, total=len(all_sid)):
                cur_part = df_sort_outk_selected[df_sort_outk_selected['sid']==sid].sort_values('dr_label', ascending=False)
                # cur_part = cur_part[(cur_part['dr_label']>=up_quantiles[sid]) | (cur_part['dr_label']<=low_quantiles[sid])].sort_values('dr_label', ascending=False)
                up_part = cur_part[(cur_part['dr_label']>=up_quantiles[sid])]
                low_part = cur_part[(cur_part['dr_label']<=low_quantiles[sid])]
                # rows_left = [r for _,r in low_part.iterrows()]
                for _, row_out in up_part.iterrows():
                    rows_left = list(filter(lambda x: x is not None, [r if (row_out['dr_score']*row_out['dr_label'] - r['dr_score']*r['dr_label']) > 0 else None 
                                                    for _,r in low_part.iterrows()]))
                    rows_cur = [row_out] * len(rows_left)
                    tag_list = [False] * len(rows_left)
                    mode_list = [mode] * len(rows_left)
                    rows_pair.extend(list(map(self._assign_val, rows_cur, rows_left, tag_list, mode_list)))

                # length = cur_part.shape[0]
                # start_idx = cur_part.iloc[0].name
                # if (cur_part.head(1)['dr_label'].values - cur_part.tail(1)['dr_label'].values)[0] >= 0:
                #     for idx, row_out in cur_part.iterrows():
                #         cur_idx = idx
                #         rows_left = list(filter(lambda x: x is not None, [r if (row_out['dr_score']*row_out['dr_label'] - r['dr_score']*r['dr_label']) >= 0 else None 
                #                                     for _,r in cur_part.iloc[cur_idx - start_idx + 1:length].iterrows()]))
                #         tag_list = [False] * len(rows_left)
                #         rows_cur = [row_out] * len(rows_left)
                #         mode_list = [mode] * len(rows_left)

                #         rows_pair.extend(list(map(self._assign_val, rows_cur, rows_left, tag_list, mode_list)))
                # else:
                #     continue


        # assign data into DataFrame
        rows_pair = np.array(rows_pair, dtype=object)
        for i, col_name in enumerate(df_pair.columns):
            df_pair[col_name] = rows_pair[:,i]
        #df_pair = df_pair[df_pair['rel_diff']>=self.threshold * self.lamda]
        df_pair = df_pair.sort_values(by=['sid'], ascending=[True])
        df_pair.reset_index(drop=True, inplace=True) 

        return df_pair

    @staticmethod
    def _assign_val(row_out, row_in, isTopK=True, mode='train'):
        if isTopK==True:
            tag = 1
        else:
            tag = 0
        if mode=='train':
            row_new = [ row_out['sid'],
                        tag,
                        (row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'] - row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']),
                        # (row_out['normalized_var'] * row_in['normalized_var']) * (row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'] - row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']), 
                        row_out['did'],
                        row_in['did']]
        elif mode=='vali':
            row_new = [ row_out['sid'], 
                        tag,
                        (row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'] - row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']),
                        # (row_out['normalized_var'] * row_in['normalized_var']) * (row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'] - row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']), 
                        row_out['did'], row_out['feature'], 
                        row_in['did'], row_in['feature']]           
        return row_new


    @staticmethod
    def _assign_val2(row_out, row_in, isTopK=True, mode='train'):
        if isTopK==True:
            tag = 1
        else:
            tag = 0
        if mode=='train':
            row_new = [ row_out['sid'],
                        tag,
                        1,
                        # np.log(1 + row_out['dr_score']*row_out['dr_label']*row_out['dr_weight']) - np.log(1 + row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']),
                        # row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'],
                        # (row_out['normalized_var'] * row_in['normalized_var']) * (row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'] - row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']), 
                        row_out['did'],
                        row_in['did']]
        elif mode=='vali':
            row_new = [ row_out['sid'], 
                        tag,
                        1,
                        # np.log(1 + row_out['dr_score']*row_out['dr_label']*row_out['dr_weight']) - np.log(1 + row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']),
                        # row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'],
                        # (row_out['normalized_var'] * row_in['normalized_var']) * (row_out['dr_score']*row_out['dr_label']*row_out['dr_weight'] - row_in['dr_score']*row_in['dr_label']*row_in['dr_weight']), 
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
            ips_early_stopping = EarlyStopping(self.fout, self.patience, verbose=True) 
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
                ndcg_val = eval_result['NDCG'][self.topK]
                mrr_val = eval_result['MRR'][self.topK]
                precision_val = eval_result['Precision'][self.topK]
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
                    wt = row['ips_weight']
                    lamda = 1
                    p_label = row['isClick']
                else:
                    # wt = row['estimate_score']
                    wt = 1
                    lamda = self.lamda
                    # p_label = 1
                    p_label = row['estimate_score']
                #wt = row['ips_weight'] * (row['isClick'] - row['estimate_label']) + row['estimate_label']
                label_train.append(p_label)
                dr_score_train.append(wt)
                lamda_train.append(lamda)
            dr_score_vali = []
            lamda_vali = []
            label_vali = []
            for _, row in vali_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']
                    lamda = 1 
                    p_label = row['isClick']
                else:
                    # wt = row['estimate_score']
                    wt = 1
                    lamda = self.lamda
                    # p_label = 1
                    p_label = row['estimate_score']
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
            vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair_topk.json')
            #vali_log_out_topK = vali_log[vali_log['rankPosition'] >= self.topK]
            ips_vali_tool = Vali_Evaluator(vali_log_topK, 
                                        self.eval_positions, 
                                        use_cuda=self.use_cuda, 
                                        with_weight='ips',
                                        pair_wise=True, pair_df=vali_log_pair)

            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            train_log_topK = train_log[train_log['rankPosition'] < self.topK]
            train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair_topk.json')
            # train_log_pair = train_log_pair[train_log_pair['click_diff']==1]
            #train_log_out_topK = train_log[train_log['rankPosition'] >= self.topK]

            # if self.sparse_tag:
            #     input_train = Wrap_Dataset_clickLog(doc_tensor=get_sparse_feature(train_log_topK['feature']), 
            #                                     click_tensor = torch.Tensor(train_log_topK['isClick']), 
            #                                     ips_tensor = torch.Tensor(train_log_topK['ips_weight_for_train']))
            # else:
            #     input_train = Wrap_Dataset_clickLog(doc_tensor=torch.Tensor(train_log_topK['feature']), 
            #                                     click_tensor = torch.Tensor(train_log_topK['isClick']), 
            #                                     ips_tensor = torch.Tensor(train_log_topK['ips_weight_for_train']))
            input_train = Wrap_Dataset_Pairwise2(train_log_pair['pos_did'].to_list(), 
                                                    train_log_pair['neg_did'].to_list(),
                                                    train_log_pair['rel_diff'].to_list(),
                                                    train_log_topK,
                                                    sparse_tag=self.sparse_tag)
            ips_input_train_loader = DataLoader(input_train, 
                                            batch_size=self.pre_bs, 
                                            shuffle=True)
            if self.continue_tag == True:
                # ips_model = rank_model2(self.in_size, self.hidden_size, self.drop_out)
                # ips_model = rank_model(self.in_size, self.hidden_size, self.drop_out)
                ips_model = rank_model_pro(self.in_size, self.hidden_size, self.drop_out)
                if self.use_cuda:
                    ips_model = ips_model.cuda()
                # predict estimate label for train and vali
                ips_model.load_state_dict(torch.load('{}_pretrainIPS.pt'.format(self.fout)))
                # ips_model.eval()
                # # evaluate model on test set
                # test_result = self.test_tool.evaluate(ips_model)
                # test_result.to_json('{}_Pretrain_result.json'.format(self.fout), indent=4)
            else:
                ips_model = rank_model(self.in_size, self.hidden_size, self.drop_out)
                # ips_model = rank_model2(self.in_size, self.hidden_size, self.drop_out)
                # ips_model = rank_model2_2(self.in_size, self.hidden_size, self.drop_out)
                ips_model.weight_init()
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
                ips_early_stopping = EarlyStopping(fp=self.fout, patience = 20, verbose=True) 

                dur = []
                for epoch in range(self.epoch):
                    if epoch >= 3:
                        t0 = time.time()
                    
                    loss_log = []
                    ips_model.train()
                    for _id, batch in enumerate(ips_input_train_loader):
                        ips_optim.zero_grad()
                        # BCE_lossfunc = BCELoss(weight=batch[2])
                        # MSE_lossfunc = Unbias_MSELoss_reg(weight=batch[2],C=self.C)
                        # BPR_lossfunc = BPRLoss_var_reg(weight=batch[2],C=self.C)
                        BPR_lossfunc = BPRLoss(weight=batch[2])
                        
                        # var_val, output_3 = ips_model(batch[0])
                        # output_3 = output_3.view(batch[0].size(0))
                        # train_loss = MSE_lossfunc(var_val, output_3, batch[1])

                        # var_posi,output_posi = ips_model(batch[0])
                        # var_nega, output_nega = ips_model(batch[1])
                        # output_posi = output_posi.view(batch[0].size(0))
                        # output_nega = output_nega.view(batch[1].size(0))
                        # train_loss = BPR_lossfunc(var_posi, var_nega, output_posi, output_nega)

                        output_posi = ips_model(batch[0])
                        output_nega = ips_model(batch[1])
                        output_posi = output_posi.view(batch[0].size(0))
                        output_nega = output_nega.view(batch[1].size(0))
                        train_loss = BPR_lossfunc(output_posi, output_nega)

                        # out_posi_list = ips_model(batch[0])
                        # out_nega_list = ips_model(batch[1])
                        # # BPR_lossfunc = BPRLoss(weight=batch[2])
                        # train_loss_1 = BPR_lossfunc(out_posi_list[1],out_nega_list[1])
                        # train_loss_1.backward()
                        # for i in range(2,len(out_posi_list)):
                        #     # BPR_lossfunc = BPRLoss(weight=batch[2])
                        #     train_loss = BPR_lossfunc(out_posi_list[i],out_nega_list[i])
                        #     train_loss.backward()

                        train_loss.backward()
                        ips_optim.step()
                        loss_log.append(train_loss.item())

                    # evaluate performance on validation click log
                    # val_loss, eval_result = ips_vali_tool.evaluate(ips_model, 
                    #                                                 loss_type=Unbias_MSELoss)
                    val_loss, eval_result = ips_vali_tool.evaluate(ips_model, 
                                                                    loss_type=BPRLoss)
                    ndcg_val = eval_result['NDCG'][self.topK]
                    mrr_val = eval_result['MRR'][self.topK]
                    precision_val = eval_result['Precision'][self.topK]

                    test_result = self.test_tool.evaluate(ips_model)
                    ndcg_tst = test_result['NDCG'][self.topK]
                    mrr_tst = test_result['MRR'][self.topK]
                    precision_tst = test_result['Precision'][self.topK]
                    map_tst = test_result['MAP'][self.topK]
                    ndcg_full_tst = test_result['NDCG_full'][self.topK]

                    # if self.schedul_mode == 'max':
                    #     ips_scheduler.step(ndcg_val)
                    # elif self.schedul_mode == 'min':
                    #     ips_scheduler.step(val_loss)

                    # ips_scheduler.step(np.mean(loss_log))

                    # if epoch >= self.epoch_start:
                    #     if self.schedul_mode == 'max':
                    #         ips_early_stopping(ndcg_val*(-1), ips_model)
                    #     elif self.schedul_mode == 'min':
                    #         ips_early_stopping(val_loss, ips_model)

                        # ips_early_stopping(ndcg_val*(-1), ips_model)
                        # ips_early_stopping(np.mean(loss_log), ips_model)
                        # ips_early_stopping(val_loss, ips_model)
                        # ips_early_stopping((ndcg_val+mrr_val)*(-1/2), ips_model)

                    # if ips_early_stopping.early_stop:
                    #     print("Early stopping")
                    #     break 
                    
                    if epoch >= 3:
                        dur.append(time.time() - t0)

                    torch.save(ips_model.state_dict(), '{}_checkpoint.pt'.format(self.fout))

                    print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Val_NDCG@10 {:.4f} | "
                            "Val_MRR@10 {:.4f}| Val_Precision@10 {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),val_loss,
                                                            ndcg_val, mrr_val, precision_val))     
                                                            
                    print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Test_NDCG@10 {:.4f} | "
                    "Test_MRR@10 {:.4f}| Test_Precision@10 {:.4f} | Test_MAP@10 {:.4f} | Test_NDCG_full {:.4f} |". format(epoch, np.mean(dur), np.mean(loss_log),ndcg_tst,
                                                    mrr_tst, precision_tst, map_tst, ndcg_full_tst))

                # # predict estimate label for train and vali
                # ips_model.load_state_dict(torch.load('{}_checkpoint.pt'.format(self.fout)))
                # ips_model.eval()

               
                torch.save(ips_model.state_dict(), '{}_pretrainIPS.pt'.format(self.fout))
                ips_model.eval()

                # evaluate model on test set
                test_result = self.test_tool.evaluate(ips_model)
                test_result.to_json('{}_Pretrain_result.json'.format(self.fout), indent=4)

            if self.use_cuda:              
                with torch.no_grad():
                    ips_model.eval()
                    train_predict = ips_model(torch.Tensor(train_log['feature']).cuda())
                    vali_predict = ips_model(torch.Tensor(vali_log['feature']).cuda())
                    if isinstance(train_predict, tuple):
                        train_predict = train_predict[1]
                    if isinstance(vali_predict, tuple):
                        vali_predict = vali_predict[1]
                    # train_var, train_predict = ips_model(torch.Tensor(train_log['feature']).cuda())
                    # vali_var, vali_predict = ips_model(torch.Tensor(vali_log['feature']).cuda())
                    # train_var, train_predict,_,_,_ = ips_model(torch.Tensor(train_log['feature']).cuda())
                    # vali_var, vali_predict,_,_,_ = ips_model(torch.Tensor(vali_log['feature']).cuda())
            else:
                with torch.no_grad():
                    ips_model.eval()
                    train_predict = ips_model(torch.Tensor(train_log['feature']).cpu())
                    vali_predict = ips_model(torch.Tensor(vali_log['feature']).cpu())
                    if isinstance(train_predict, tuple):
                        train_predict = train_predict[1]
                    if isinstance(vali_predict, tuple):
                        vali_predict = vali_predict[1]
                    # train_var, train_predict = ips_model(torch.Tensor(train_log['feature']).cpu())
                    # vali_var, vali_predict = ips_model(torch.Tensor(vali_log['feature']).cpu())
                    # train_var, train_predict,_,_,_ = ips_model(torch.Tensor(train_log['feature']).cpu())
                    # vali_var, vali_predict,_,_,_ = ips_model(torch.Tensor(vali_log['feature']).cpu())                
            train_predict = train_predict.cpu()
            train_predict = train_predict.view(-1).tolist()
            # train_var = train_var.cpu()
            # train_var = train_var.view(-1).tolist()
            vali_predict = vali_predict.cpu()
            vali_predict = vali_predict.view(-1).tolist()
            # vali_var = vali_var.cpu()
            # vali_var = vali_var.view(-1).tolist()
            train_log['estimate_score'] = train_predict
            # train_log['estimate_var'] = train_var
            vali_log['estimate_score'] = vali_predict
            # vali_log['estimate_var'] = vali_var
            dr_score_train = []
            lamda_train = []
            label_train = []
            for _, row in train_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']
                    # wt = 1
                    lamda = 1
                    p_label = row['isClick']
                else:
                    wt = 1
                    lamda = self.lamda
                    p_label = row['estimate_score']
                    # p_label = row['label']
                #wt = row['ips_weight'] * (row['isClick'] - row['estimate_label']) + row['estimate_label']
                label_train.append(p_label)
                dr_score_train.append(wt)
                lamda_train.append(lamda)
            dr_score_vali = []
            lamda_vali = []
            label_vali = []
            for _, row in vali_log.iterrows(): 
                if row['rankPosition']<self.topK:
                    wt = row['ips_weight']
                    lamda = 1 
                    p_label = row['isClick']
                else:
                    # wt = row['estimate_score']*self.lamda
                    wt = 1
                    lamda = self.lamda
                    # p_label = 1
                    p_label = row['estimate_score']
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
            # torch.save(ips_model.state_dict(), '{}_pretrainIPS.pt'.format(self.fout))

        return train_log, vali_log


if __name__=='__main__':
    parser = config_param_parser()
    args = parser.parse_args()
    dr = Doubly_Robust(args)
    dr.train()

    