import argparse


########################################## supervised algorithm

parser = argparse.ArgumentParser() 
parser.add_argument('-algo','--algo', default='sup', type=str , help=' sup: supervised algorithm, unsup: unsupervised algorithm')  
parser.add_argument('-t','--model_type', default='32', type=str , help='fusion,  32,  64,  128,  256')  
parser.add_argument('-a','--action', default='train', type=str,  help='train, predict, load')  # train predict load
parser.add_argument('-d','--dataset', default='50salads', type=str,  help='50salads,  gtea,  breakfast') # 50salads  gtea  breakfast
parser.add_argument('-sp','--split', default='1', type=str,  help='')
parser.add_argument('-gpu','--gpu_num', default='0', type=str,  help='')
parser.add_argument('-last','--last_epoch', default=23, type=int,  help='loading last epoch to resume training')  
parser.add_argument('-nepoc','--num_epcs', default=300, type=int,  help='')  

parser.add_argument('-bw','--bound_wind', default=16, type=int,  help='boundary window in boundary correction algorithm')  # 16 for salades 8 fro gtea 
parser.add_argument('-sw','--sim_wind', default=4, type=int,  help='similarity window size of the boundary window')  # 4 for salad and gtea (how many splits of the boundary window)

# at prediction time
parser.add_argument('-asth','--auto_smooth',  action='store_true',  help='automatic specify smoothing window') 
parser.add_argument('-sthw','--smth_wind', default=80, type=int,  help='smoothing window sise in case of manual specification')  #  # 80 for salad 4 for gtea  4 for gtea
parser.add_argument('-bndin','--bound_in_pred', action='store_true',  help='use boundary correction after the voting at test time inside the fusionmodel')  #  
parser.add_argument('-bndaft','--bound_aft_pred', action='store_true',  help='use boundary correction after prediction at test time')  # Smoothing without smth_wind, (max distance between boundaries)/10


# encoders best check points and fusion model best check points
parser.add_argument('-ck32','--checkpoint32', default=94, type=int,help='best checkpoint of encoder 32')  
parser.add_argument('-ck64','--checkpoint64', default=48, type=int,help='best checkpoint of encoder 64')   
parser.add_argument('-ck128','--checkpoint128', default=46, type=int,help='best checkpoint of encoder 128')  
parser.add_argument('-ck256','--checkpoint256', default=62, type=int,help='best checkpoint of encoder 256') 

parser.add_argument('-ckfuse','--checkpointfuse', default=62, type=int,help='best checkpoint of the fusion model')  # When loading last epoch to continue training or for prediction


########################################## unsupervised algorithm

#unsupervised threshold 
parser.add_argument('-unsthraut','--unsup_thresh_auto',  action='store_true',help='in case of unsupervised algorthm, if want to use automatic threshols interval between boundaries set this to true')  

args = parser.parse_args() 
