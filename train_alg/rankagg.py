import torch
import pandas as pd
import numpy as np
from torch.nn import BCELoss,BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import sys
import time
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser
from utils.data_wrapper import Wrap_Dataset_clickLog, Wrap_Dataset_fullInfo, Wrap_Dataset_Pairwise, Wrap_Dataset_Pairwise2
from utils.get_sparse import get_sparse_feature
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator
from models.loss_func import BPRLoss, HingeLoss,BPRLoss_log_norm,BPRLoss_var_reg


class Rankagg(Learning_Alg):
    def __init__(self, args):
        super(Rankagg, self).__init__(args)
        
    def train(self):
        t0 = time.time()
        self.test_tool, self.in_size = super(Rankagg, self)._load_test_and_get_tool()
        t1 = time.time()
        self.vali_tool = self._load_vali_and_get_tool()
        t2 = time.time()
        input_train_loader = self._load_and_wrap_train()
        input_train_log = self._load_train_heckman()
        t3 = time.time()
        print('load test time:', t1-t0)
        print('load vali time:', t2-t1)
        print('load train time:', t3-t2)
        self.model, self.optim, self.scheduler, self.early_stopping = super(Rankagg, self)._init_train_env()
        print('Train IPS...')
        super(Rankagg, self)._train_iteration(input_train_loader)
        print('Train Heckman...')
        self._train_heckman(input_train_log)
        print('Aggregation and Evaluate...')
        self._test_and_save()


    def _test_and_save(self):
        # load best model
        self.x_coeff = np.loadtxt('{}_params.txt'.format(self.fout))
        self.x_coeff = self.x_coeff[0:-1]
        self.model.load_state_dict(torch.load('{}_checkpoint.pt'.format(self.fout)))
        

        # evaluate model on test set
        test_result = self.test_tool.evaluate_agg(self.model, self.x_coeff)

        # save model and result as given path
        #print('test_loss:', test_loss)
        torch.save(self.model.state_dict(), '{}_model.pt'.format(self.fout))
        test_result.to_json('{}_result.json'.format(self.fout), indent=4)


    def _train_heckman(self, input_train_log):
        Y_sel = input_train_log['sel_tag']
        Y_click = input_train_log['isClick']
        X = np.array(input_train_log['feature'].values.tolist())
        gamma = self._probit(Y_sel, X).T
        lambda_ = self._inverse_mills(np.matmul(X, gamma))
        params = self._probit(Y_click, np.append(X, lambda_.reshape(-1,1), 1))
        print(params.shape)
        print(params)
        self.lam_coeff= params[0][-1]
        self.x_coeff= params[0][0:-1]
        # np.savetxt('{}_lam_coeff.txt'.format(self.fout),self.lam_coeff)
        # np.savetxt('{}_x_coeff.txt'.format(self.fout),self.x_coeff)
        np.savetxt('{}_params.txt'.format(self.fout),params)
        
    def _probit(self, Y, X):   
        # clf = LogisticRegression(solver='lbfgs',C=self.weight_decay, max_iter=self.epoch, verbose=1, random_state=41).fit(X, Y)
        # clf = LogisticRegression(solver='lbfgs', verbose=1, C=1e-4, random_state=self.randseed).fit(X, Y)
        clf = LogisticRegression(solver='lbfgs', verbose=1, random_state=self.randseed).fit(X, Y)
        return clf.coef_/1.6

    def _inverse_mills(self, val):
        return norm.pdf(val) / norm.cdf(val)

    def _train_one_batch(self, batch):
        #print('------------------------')
        self.optim.zero_grad()
        if self.pairwise:
            BPR_lossfunc = BPRLoss(weight=batch[2])
            # BPR_lossfunc = BPRLoss_log_norm(weight=batch[2])
            output_posi = self.model(batch[0])
            output_nega = self.model(batch[1])
            # BPR_lossfunc = BPRLoss_var_reg(weight=batch[2],C=self.C)
            # var_posi,output_posi = self.model(batch[0])
            # var_nega, output_nega = self.model(batch[1])

            output_posi = output_posi.view(batch[0].size(0))
            output_nega = output_nega.view(batch[1].size(0))

            train_loss = BPR_lossfunc(output_posi, output_nega)
            # train_loss = BPR_lossfunc(var_posi, var_nega, output_posi, output_nega)
        else:
            BCE_lossfunc = BPRLoss_log_norm(weight=batch[2])
            output = self.model(batch[0])
            output = output.view(batch[0].size(0))
            train_loss = BCE_lossfunc(output, batch[1])
        train_loss.backward()
        self.optim.step()
        return train_loss
        
    def _load_vali_and_get_tool(self):
        if self.pairwise:
            vali_log = pd.read_json(self.fin + 'click_log/Vali_log.json')
            vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair.json')
            if self.topK > 0:
                vali_log = vali_log[vali_log['rankPosition'] < self.topK]
                vali_log_pair = pd.read_json(self.fin + 'click_log/Vali_log_pair_topk.json')
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
                train_log_pair = pd.read_json(self.fin + 'click_log/Train_log_pair_topk_sel.json')
                train_log_pair = train_log_pair[train_log_pair['tag']==1]
                # train_log_pair.loc[train_log_pair['click_diff']==0,'rel_diff']=1
                # train_log_pair.loc[train_log_pair['click_diff']==1,'rel_diff']=1
                train_log = train_log[train_log['rankPosition'] < self.topK]
            # if self.sparse_tag:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=get_sparse_feature(train_log_pair['pos_feature']),
            #                                         neg_feature=get_sparse_feature(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['rel_diff']))
            # else:
            #     input_train = Wrap_Dataset_Pairwise(posi_feature=torch.Tensor(train_log_pair['pos_feature']),
            #                                         neg_feature=torch.Tensor(train_log_pair['neg_feature']),  
            #                                         rel_diff=torch.Tensor(train_log_pair['rel_diff']))
            input_train = Wrap_Dataset_Pairwise2(train_log_pair['pos_did'].to_list(), 
                                                    train_log_pair['neg_did'].to_list(),
                                                    train_log_pair['rel_diff'].to_list(),
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

    def _load_train_heckman(self):
        train_dat = pd.read_json(self.fin + 'json_file/Train.json')
        train_dat = format_trans(train_dat)
        train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
        new_train_log = pd.merge(train_log,train_dat[['did','feature']],on='did')
        sel_tag_ls = new_train_log['rankPosition'].apply(lambda x: 1 if x<self.topK else 0)
        new_train_log['sel_tag'] = sel_tag_ls
        return new_train_log

if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = Rankagg(args)
    learner.train()