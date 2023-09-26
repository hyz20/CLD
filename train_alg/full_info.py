import sys
sys.path.append("..")
from models.base_learn_alg import Learning_Alg
from utils.arg_parser import config_param_parser

class Full_Info(Learning_Alg):
    def __init__(self, args):
        super(Full_Info, self).__init__(args)
        
    def learning(self):
        super(Full_Info, self).train()

if __name__=="__main__":
    print('Start ...')
    parser = config_param_parser()
    args = parser.parse_args()
    learner = Full_Info(args)
    learner.train()