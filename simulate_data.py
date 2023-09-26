import numpy as np
import pandas as pd
import argparse
import json
import random
import os,sys
import utils.click_model as CM
from tqdm import tqdm
from argparse import ArgumentTypeError
from utils.click_model import PositionBiasedModel
from utils.load_data import load_data_forEstimate
from utils.pairwise_trans import get_pair, get_pair_fullinfo
from utils.trans_format import format_trans
from utils.set_seed import setup_seed
from simulate.simulate_click import simulateOneSession
from simulate.estimate_ips import RandomizedPropensityEstimator
from simulate.estimate_rel import RandomizedRelevanceEstimator

# ===============================================
#
# given a dataset
# 1. estimate ips weight for each position
# 2. simulate click for a sampled query
# 3. calculate ips weight for each click session
# 4. save click log
# 5. estimate relevance for each observed doc
# 6. save relevance estimate
# 7. transform click log & label data into pairwise format
# 8. save pairwise format
#
# ===============================================

def simulate_data(file_path, file_path_out, session_num = 1e3, isLoad = False, TopK = 10, eta = 1.0, noise = 0.1): 
    train_file = file_path + 'json_file/Train.json'
    vali_file = file_path + 'json_file/Vali.json'

    output_file_path = file_path + 'click_log/'
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)

    # Load data from json file
    train_data = load_data_forEstimate(train_file)
    vali_data = load_data_forEstimate(vali_file)

    if isLoad:
        # Load a saved click model from json file
        with open(file_path_out + 'pbm.json') as fin:	
            clickmodel_dict = json.load(fin)
        pbm = CM.loadModelFromJson(clickmodel_dict)
        # Load a IPS Estimator from json file
        estimator = RandomizedPropensityEstimator()
        estimator.loadEstimatorFromFile(file_path_out + 'ips_estimator.json')
    else:
        # Init a click model
        pbm = CM.PositionBiasedModel(eta=eta, TopK=TopK, pos_click_prob=1.0, neg_click_prob=noise)
        pbm.setExamProb(eta)
        pbm.setClickProb(noise, 1.0, 1)
        pbm.outputModelJson(file_path_out + 'pbm.json')
        # Init a IPS Estimator 
        estimator = RandomizedPropensityEstimator()
        print('Estimating IPS from radomized click session...')
        estimator.estimateParametersFromModel(pbm, train_data, 10e5)
        estimator.outputEstimatorToFile(file_path_out + 'ips_estimator.json')

    print('Simulate click sessions...')

    # simulate on train data
    overallSessionLog = []
    for sid in tqdm(range(int(session_num))):
        #queryID = random.randint(0,train_data.rank_list_size - 1)
        queryID = np.random.choice(train_data.qid_lists, 1, replace=True)[0]
        query_list = train_data.query_lists[queryID]
        oneSessionLog = simulateOneSession(pbm, query_list)
        ips_list = estimator.getPropensityForOneList([d['isClick'] for d in oneSessionLog])
        ips_for_train_list = estimator.getPropensityForOneList_ForTrain([d['isClick'] for d in oneSessionLog])
        for index, doc in enumerate(oneSessionLog):
            doc['ips_weight'] = ips_list[index]
            doc['ips_weight_for_train'] = ips_for_train_list[index]
            doc['sid'] = sid
        overallSessionLog.extend(oneSessionLog)

    with open(output_file_path + 'Train_log.json', 'w') as fout:
        fout.write(json.dumps(overallSessionLog))

    # simulate on vali data
    overallSessionLog = []
    vali_session_num = session_num * (vali_data.data_size / train_data.data_size)
    for sid in tqdm(range(int(vali_session_num))):
        #queryID = random.randint(0,vali_data.rank_list_size - 1)
        queryID = np.random.choice(vali_data.qid_lists, 1, replace=True)[0]
        query_list = vali_data.query_lists[queryID]
        oneSessionLog = simulateOneSession(pbm, query_list)
        ips_list = estimator.getPropensityForOneList([d['isClick'] for d in oneSessionLog])
        ips_for_train_list = estimator.getPropensityForOneList_ForTrain([d['isClick'] for d in oneSessionLog])
        for index, doc in enumerate(oneSessionLog):
            doc['ips_weight'] = ips_list[index]
            doc['ips_weight_for_train'] = ips_for_train_list[index]
            doc['sid'] = sid
        overallSessionLog.extend(oneSessionLog)

    with open(output_file_path + 'Vali_log.json', 'w') as fout:
        fout.write(json.dumps(overallSessionLog))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', required=True)
    parser.add_argument('--fp2', required=True)
    parser.add_argument('--session_num', type=float, default=1e3)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--TopK', type=int, default=10)
    parser.add_argument('--isLoad',type=str2bool, nargs='?', default=False)

    args = parser.parse_args()
    setup_seed(41)
    simulate_data(args.fp, args.fp2, session_num=args.session_num, isLoad=args.isLoad, TopK=args.TopK, eta=args.eta, noise=args.noise)
    

