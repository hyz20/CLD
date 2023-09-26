import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sys
import time
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2
from utils.get_sparse import get_sparse_feature
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator



class Naive(Learning_Alg):
    def __init__(self, args):
        super(Naive, self).__init__(args)
        
    def train(self):
        t0 = time.time()
        self.test_tool, self.in_size = super(Naive, self)._load_test_and_get_tool()
        print(self.in_size)
        t1 = time.time()
        self.vali_tool = self._load_vali_and_get_tool()
        t2 = time.time()
        input_train_loader = self._load_and_wrap_train()
        t3 = time.time()
        print('load test time:', t1-t0)
        print('load vali time:', t2-t1)
        print('load train time:', t3-t2)

        self.model, self.optim, self.scheduler, self.early_stopping = super(Naive, self)._init_train_env()
        super(Naive, self)._train_iteration(input_train_loader)
        super(Naive, self)._test_and_save()
    
        
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
    
    def _load_and_wrap_train(self):
        if self.pairwise:
            train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair.json')
            # train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            train_log = pd.read_json(self.fin + 'json_file/Train.json')
            train_log = format_trans(train_log)
            if self.topK > 0:
                train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair_topk_sel.json')
                train_log_pair = train_log_pair[train_log_pair['tag']==1]
                train_log = train_log[train_log['rankPosition'] < self.topK]
            # if self.sparse_tag:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair['pos_feature']),
            #                                         neg_feature=get_sparse_feature(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['click_diff']))
            # else:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair['pos_feature']),
            #                                         neg_feature=torch.Tensor(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['click_diff']))
            input_train = Wrap_Dataset_Pairwise2(train_log_pair['pos_did'].to_list(), 
                                                    train_log_pair['neg_did'].to_list(),
                                                    train_log_pair['click_diff'].to_list(),
                                                    train_log,
                                                    sparse_tag=self.sparse_tag)
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        else:
            train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
            if self.topK > 0:
                train_log = train_log[train_log['rankPosition'] < self.topK]
            
            input_train = Wrap_Dataset_fullInfo(doc_tensor=torch.Tensor(train_log['feature']), 
                                            label_tensor = torch.Tensor(train_log['isClick']))
            input_train_loader = DataLoader(input_train, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return input_train_loader


if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = Naive(args)
    learner.train()