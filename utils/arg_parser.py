import argparse
from argparse import ArgumentTypeError

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def config_param_parser():
    parser = argparse.ArgumentParser(description="Experiment Configures and Model Parameters")
    
    # Experiment Configures
    parser.add_argument('--fin', required=True)
    parser.add_argument('--fout',required=True)
    parser.add_argument('--use_cuda', type=str2bool, nargs='?', default=True)
    #parser.add_argument('--use_full_info', type=str2bool, nargs='?', default=False)
    #parser.add_argument('--use_naive', type=str2bool, nargs='?', default=False)
    parser.add_argument('--train_alg', type=str, choices=['ips','ips_sel','ips_sel_co', 'ips_tobit', 'ips_tobit2', 'naive', 'full_info', 'dr', 'dm', 'dr_pro', 'dr_full','unify','heckman','pijd', 'rankagg'], required=True)
    parser.add_argument('--pairwise', type=str2bool, nargs='?', default=True)
    parser.add_argument('--eval_positions', type=int, nargs='*',default=[1,2,3,5,10,20])
    parser.add_argument('--topK', type=int , default= 10)
    parser.add_argument('--randseed', type=int , default=41)
    parser.add_argument('--sparse_tag', type=str2bool, nargs='?', default=False)
    parser.add_argument('--continue_tag', type=str2bool, nargs='?', default=False)
    
    # Model Parameters
    parser.add_argument('--ratio', type=float, default=1, help="contrall the frac of unobserve pair")
    parser.add_argument('--train_method', type=str, default='unify', choices=['unify', 'twostep', 'joint'])
    parser.add_argument('--rho', type=float, default=0.5, help="contrall correlation")
    parser.add_argument('--C', type=float, default=1, help="contrall the weight of variance regularization")
    parser.add_argument('--rel_estimator', type=str, default='ips', choices=['ips','random'], help='this arg only used for DR and DM')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--schedul_mode', type=str, default='min', choices=['min', 'max'])
    parser.add_argument('--lamda', type=float, default=1.0, help='contral the contribution for data out of TopK, only used for DR')
    parser.add_argument('--threshold', type=float, default=0.05, help='contral the minimaize value to form pair, only used for DR and DM')
    parser.add_argument('--session_num', type=float, default=1e3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--drop_out', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--pre_wd', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pre_bs',type=int,default=128)
    parser.add_argument('--patience', type=int, default=20, help="waiting patience for early stop")
    parser.add_argument('--epoch_start', type=int, default=0, help="epoch when early stop start")
    parser.add_argument('--hidden_size', type=int, default=32, help="hidden size of latent layer in NN")
    return parser


if __name__=="__main__":
    parser = config_param_parser()
    args = parser.parse_args()
    print(args)
    