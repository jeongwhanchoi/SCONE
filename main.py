from SELFRec import SELFRec
import os
import torch
import numpy as np
import argparse
import random
import time



def main(args):
    
    randomSeed = 2023
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(randomSeed)
    np.random.seed(randomSeed)

    GPU_NUM = args.device
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU

    s = time.time()
    print(args)
    rec = SELFRec(args)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='douban', help='dataset name')
    parser.add_argument('--model', type=str, default='SCONE', help='model name')
    parser.add_argument('--model_type', type=str, default='graph', help=' ')
    parser.add_argument('--embedding_size', type=int, default=64, help=' ')
    parser.add_argument('--score_dim', type=str, default='64,128', help=' ') ###
    parser.add_argument('--n_layer', type=int, default=2, help=' ')
    parser.add_argument('--NS', type=str2bool, default=True, help="Activate nice mode.")
    parser.add_argument('--CL', type=str2bool, default=True, help="Activate nice mode.")
    parser.add_argument('--save_model', type=str2bool, default=False, help="Activate nice mode.")

    #Evaluation
    parser.add_argument('--topN', type=str, default='10,20', help=' ') ###

    #Training
    parser.add_argument('--batch_size', type=int, default=2048, help=' ')
    parser.add_argument('--epoch', type=int, default=200, help=' ')
    parser.add_argument('--lr', type=float, default=1e-03, help=' ')
    parser.add_argument('--simgcl_noise', type=float, default=0.2, help=' ')
    parser.add_argument('--lr_score', type=float, default=1e-03, help=' ')
    parser.add_argument('--reg', type=float, default=1e-04, help=' ')
    parser.add_argument('--lambda_cl', type=float, default=0.5, help=' ')

    parser.add_argument('--noise_min', type=float, default=0.01, help=' ')
    parser.add_argument('--noise_max', type=float, default=50, help=' ')
    parser.add_argument('--T', type=int, default=100, help=' ')
    parser.add_argument('--sampling_ratio', type=float, default=0.1, help=' ')
    parser.add_argument('--weight', type=float, default=0.9, help=' ')
    parser.add_argument('--sde_type', type=str, default='vesde', help=' ')
    parser.add_argument('--output_dir', type=str, default='./SCONE_exp/', help=' ')
    parser.add_argument('--device', default=0, type=int, help='the gpu to use')

    args = parser.parse_args()
    main(args)
