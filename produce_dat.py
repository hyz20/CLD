import pandas as pd
import numpy as np

train_log_pair = pd.read_json('../datasets/WEB10K/' + 'click_log/Train_log_pair_topk.json')
train_log_pair_ob = train_log_pair[train_log_pair['tag']==1]
train_log_pair_unob_all = train_log_pair[train_log_pair['tag']==0]
train_log_pair_unob = train_log_pair_unob_all.sample(n = int(train_log_pair_ob.shape[0]*1.0), replace=False)
print(train_log_pair_ob.shape)
print(train_log_pair_unob.shape)
train_log_pair = pd.concat([train_log_pair_ob, train_log_pair_unob])
train_log_pair.to_json('../datasets/WEB10K/' + 'click_log/Train_log_pair_CLD.json')