import pandas as pd
import numpy as np
import argparse
from utils.pairwise_trans import get_pair, get_pair_fullinfo, get_pair_sel
from utils.trans_format import format_trans
from argparse import ArgumentTypeError

def transform_to_pairwise(file_path, in_type='log', TopK=0):
    if in_type == 'log':
        if TopK == 0: # dont use topK
            print("use ALL log")
            # transform click log into pairwise format
            print('Transform click log into pairwise format...')
            df_session = pd.read_json(file_path + 'click_log/Train_log.json')
            df_pair = get_pair(df_session, mode='train')
            df_pair.to_json(file_path + 'click_log/Train_log_pair.json')

            df_session = pd.read_json(file_path + 'click_log/Vali_log.json')
            df_pair = get_pair(df_session, mode='vali')
            df_pair.to_json(file_path + 'click_log/Vali_log_pair.json')

        elif TopK > 0: # use TopK
            print("use TOP_K log")
            # transform click log into pairwise format
            print('Transform click log into pairwise format...')
            df_session = pd.read_json(file_path + 'click_log/Train_log.json')
            df_session['isSelect'] = df_session['rankPosition'].apply(lambda x: 1 if x<TopK else 0)
            # df_pair = get_pair(df_session[df_session['rankPosition'] < TopK], mode='train')
            df_pair = get_pair_sel(df_session, mode='train',topk=TopK)
            df_pair.to_json(file_path + 'click_log/Train_log_pair_topk.json')
            df_pair_topk = df_pair[df_pair['tag']==1]
            df_pair_topk.to_json(file_path + 'click_log/Train_log_pair_topk_sel.json')

            df_session = pd.read_json(file_path + 'click_log/Vali_log.json')
            df_session['isSelect'] = df_session['rankPosition'].apply(lambda x: 1 if x<TopK else 0)
            df_pair = get_pair(df_session[df_session['rankPosition'] < TopK], mode='vali')
            # df_pair = get_pair_sel(df_session, mode='vali',topk=TopK)
            df_pair.to_json(file_path + 'click_log/Vali_log_pair_topk.json')
            
        else:
            raise ArgumentTypeError('invalid TopK value.')

    elif in_type == 'label':
        # transform labeled data into pairwise format
        print('Transform labeled data into pairwise format')
        df_labeled = pd.read_json(file_path + 'json_file/Train.json')
        df_labeled = format_trans(df_labeled)
        df_pair = get_pair_fullinfo(df_labeled, mode='train')
        df_pair.to_json(file_path + 'json_file/Train_pair.json')

        df_labeled = pd.read_json(file_path + 'json_file/Vali.json')
        df_labeled = format_trans(df_labeled)
        df_pair = get_pair_fullinfo(df_labeled, mode='vali')
        df_pair.to_json(file_path + 'json_file/Vali_pair.json')


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', required=True)
    parser.add_argument('--TopK', type=int, default=10)
    parser.add_argument('--in_type', type=str, default='log', choices=['log', 'label'])

    args = parser.parse_args()
    transform_to_pairwise(args.fp, TopK = args.TopK, in_type = args.in_type)