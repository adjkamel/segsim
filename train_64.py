import torch
 
from model_64 import *
from batch_gen import BatchGenerator
from eval import func_eval
import argparse
import os
import argparse
import numpy as np
import random
from arguments import  args
 
def train_64():
    device = torch.device("cuda:"+args.gpu_num if torch.cuda.is_available() else "cpu")
    seed = 19980125 # my birthday, :)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    model_type='64'  #32  #64  #128  #224 # fusion
    
    
    num_epochs = args.num_epcs #best 50
    lr = 0.0005  
    num_layers = 10
    num_f_maps = 64
    features_dim = 2048
    bz = 1
    channel_mask_rate = 0.3
    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2
        num_classes = 19
    # To prevent over-fitting for GTEA. Early stopping & large dropout rate
    if args.dataset == "gtea":
        channel_mask_rate = 0.5
        num_classes = 11
    if args.dataset == 'breakfast':
        lr = 0.0001
        num_classes = 48


    '''
    vid_list_file = "/home/aofan/action_seg/gtea/splits/train.split1.bundle"
    vid_list_file_tst = "/home/aofan/action_seg/gtea/splits/test.split1.bundle"
    features_path = "/home/aofan/action_seg/gtea/features/"
    gt_path = "/home/aofan/action_seg/gtea/groundTruth/"
    mapping_file = "/home/aofan/action_seg/gtea/mapping.txt"
    model_dir = "/home/aofan/gtea_s11_64/"
    results_dir = "/home/aofan/action_seg/last_dance_smoothing/"+args.split
    '''



    vid_list_file = args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = args.dataset+"/features/"
    gt_path = args.dataset+"/groundTruth/"
    mapping_file = args.dataset+"/mapping.txt"
    model_dir =  args.dataset+"/checkpoints/split"+args.split+ "/" +model_type+"/"
    results_dir = args.dataset+"/results/split"+ args.split+ "/" +model_type+"/"

    
    


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    index2label = dict()
    for k,v in actions_dict.items():
        index2label[v] = k



    trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(vid_list_file)

        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen_tst.read_data(vid_list_file_tst)

        trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)


    if args.action == "load":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(vid_list_file)

        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen_tst.read_data(vid_list_file_tst)

        trainer.load(model_dir, batch_gen, num_epochs, bz, lr, args.last_epoch ,batch_gen_tst)

    if args.action == "predict":
        for i in range(4,100):
            print('checkpoint',i)
            num_epochs=i
            batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
            batch_gen_tst.read_data(vid_list_file_tst)
            trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

 