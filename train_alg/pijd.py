import pandas as pd
import numpy as np
import sys
import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser
from utils.trans_format import format_trans
from models.evaluate import Test_Evaluator, Vali_Evaluator, Vali_Fullinfo_Evaluator

class PIJD(Learning_Alg):
    def __init__(self, args):
        super(PIJD, self).__init__(args)
    
    def train(self):
        print('Load data...')
        self.test_tool, self.in_size = super(PIJD, self)._load_test_and_get_tool()
        self.vali_tool = super(PIJD, self)._load_vali_and_get_tool()
        input_train_loader = self._load_and_wrap_train()
        print('Train PIJD model...')
        # self.model, self.optim, self.scheduler, self.early_stopping = super(IPS, self)._init_train_env()
        self._train_iteration(input_train_loader)
        print('Evaluate PIJD model...')
        self._test_and_save()
        print('Complete...')

    def _load_and_wrap_train(self):
        train_dat = pd.read_json(self.fin + 'json_file/Train.json')
        train_dat = format_trans(train_dat)
        train_log = pd.read_json(self.fin + 'click_log/Train_log.json')
        new_train_log = pd.merge(train_log,train_dat[['did','feature']],on='did')
        sel_tag_ls = new_train_log['rankPosition'].apply(lambda x: 1 if x<self.topK else 0)
        new_train_log['sel_tag'] = sel_tag_ls
        return new_train_log

    def _train_iteration(self, input_train_loader):
        Y_sel = input_train_loader['sel_tag']
        Y_click = input_train_loader['isClick']
        X = np.array(input_train_loader['feature'].values.tolist())
        rank_p = np.array(input_train_loader['rankPosition'].values.tolist())
        X_rp = np.append(X, rank_p.reshape(-1,1), 1)
        gamma = self._probit(Y_sel, X_rp).T
        lambda_ = self._inverse_mills(np.matmul(X_rp, gamma))

        # print(np.argwhere(np.isnan(X)))
        # print(np.argwhere(np.isnan(lambda_)))
        nan_idx = np.argwhere(np.isnan(lambda_))
        lambda_[nan_idx[:,0],nan_idx[:,1]] = 0
        inf_idx = np.argwhere(lambda_==np.inf)
        lambda_[inf_idx[:,0],inf_idx[:,1]] = 1e6
        # print(np.argwhere(np.isnan(lambda_)))
        # print(np.argwhere(lambda_==np.inf))
        params = self._probit(Y_click, np.append(X, lambda_.reshape(-1,1), 1))
        print(params.shape)
        print(params)
        self.lam_coeff= params[0][-1]
        self.x_coeff= params[0][0:-1]
        # np.savetxt('{}_lam_coeff.txt'.format(self.fout),self.lam_coeff)
        # np.savetxt('{}_x_coeff.txt'.format(self.fout),self.x_coeff)
        np.savetxt('{}_params.txt'.format(self.fout),params)
        
    def _test_and_save(self):
        test_result = self.test_tool.evaluate_linear(self.x_coeff)
        test_result.to_json('{}_result.json'.format(self.fout), indent=4)

    def _probit(self, Y, X):   
        clf = LogisticRegression(solver='lbfgs', C=5e-3, max_iter=self.epoch, verbose=1, random_state=self.randseed).fit(X, Y)
        return clf.coef_/1.6

    def _inverse_mills(self, val):
        return norm.pdf(val) / norm.cdf(val)

    