from numpy.core.fromnumeric import argsort
from train_alg.ips import IPS
from train_alg.ips_sel import IPS_sel
from train_alg.ips_sel_co import IPS_sel_co
from train_alg.naive import Naive
from train_alg.full_info import Full_Info
from train_alg.doubly_robust import Doubly_Robust
from train_alg.doubly_robust_pro import Doubly_Robust_pro
from train_alg.doubly_robust_full import Doubly_Robust_Full
from train_alg.direct_method import Direct_Method
from train_alg.unify import Unify
from train_alg.heckman import Heckman
from train_alg.pijd import PIJD
from train_alg.rankagg import Rankagg
from train_alg.ips_tobit import IPS_tobit
from train_alg.ips_tobit2 import IPS_tobit2
from utils.set_seed import setup_seed
from utils.arg_parser import config_param_parser
import torch
import warnings
import os

warnings.filterwarnings("ignore")

def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #torch.cuda.set_device(1)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = config_param_parser()
    args = parser.parse_args()
    if args.train_alg == 'full_info':
        learner = Full_Info(args)
    elif args.train_alg == 'ips':
        learner = IPS(args)
    elif args.train_alg == 'ips_sel':
        learner = IPS_sel(args)
    elif args.train_alg == 'ips_sel_co':
        learner = IPS_sel_co(args)
    elif args.train_alg == 'naive':
        learner = Naive(args)
    elif args.train_alg == 'dr':
        learner = Doubly_Robust(args)
    elif args.train_alg == 'dr_full':
        learner = Doubly_Robust_Full(args)
    elif args.train_alg == 'dr_pro':
        learner = Doubly_Robust_pro(args)
    elif args.train_alg == 'dm':
        learner = Direct_Method(args)
    elif args.train_alg =='unify':
        learner = Unify(args)
    elif args.train_alg =='heckman':
        learner = Heckman(args)
    elif args.train_alg =='pijd':
        learner = PIJD(args)
    elif args.train_alg =='rankagg':
        learner = Rankagg(args)  
    elif args.train_alg == 'ips_tobit':
        learner = IPS_tobit(args) 
    elif args.train_alg == 'ips_tobit2':
        learner = IPS_tobit2(args) 
    setup_seed(args.randseed)
    learner.train()


if __name__=="__main__":
    print('Start ...')
    main()
    print('End ...')