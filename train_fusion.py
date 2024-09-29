import torch 
 
from model_fusion import *
from batch_gen import BatchGenerator


import os
import numpy as np 
import random
from arguments import  args



def train_fusion():

    torch.set_num_threads(4)
    device = torch.device("cuda:"+args.gpu_num if torch.cuda.is_available() else "cpu")
    seed = 19980125 # my birthday, :)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

    model_type='fusion' 
    
    num_epochs = args.num_epcs # change it in the fusion_model.py not here
    bz = 1
    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        channel_mask_rate = 0.3
        sample_rate = 2
        lr = 0.0005
    # To prevent over-fitting for GTEA. Early stopping & large dropout rate
    if args.dataset == "gtea":
        channel_mask_rate = 0.9
        lr = 0.001
    if args.dataset == 'breakfast':
        channel_mask_rate = 0.3
        lr = 0.0001




    vid_list_file = args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = args.dataset+"/features/"
    gt_path = args.dataset+"/groundTruth/"
    mapping_file = args.dataset+"/mapping.txt"
    model_dir   =  args.dataset+"/checkpoints/split"+args.split+ "/" +model_type+"/"
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
    num_classes = len(actions_dict)

    
    trainer = Trainer( 2, 2,  num_classes, channel_mask_rate)

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

        batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen_tst.read_data(vid_list_file_tst)
        trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, args.checkpointfuse, actions_dict, sample_rate)




