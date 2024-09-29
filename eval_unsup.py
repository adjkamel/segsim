import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim

import torch
import time
from batch_gen import BatchGenerator

import os
import sys
import random
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from sklearn.cluster import KMeans, AgglomerativeClustering

from arguments import  args


device = torch.device("cuda:"+ args.gpu_num if torch.cuda.is_available() else "cpu")

######################### initlai clustering

sample_rate = 1
if args.dataset == "50salads":
        sample_rate = 2
        num_classes=19
    # To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    num_classes=11
if args.dataset == 'breakfast':
    lr = 0.0001
    num_classes=48



n_clusters=num_classes
random_state=42
init='k-means++'
n_init=20
# init_cluster_window=

use_conv=True
feat_dim=64
sim_wind=1  # has no influence


split_arg='1'
###########################################



def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content
 
 
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends
 
 
def levenstein(p, y, norm=False):
    m_row = len(p)    
    
    n_col = len(y)
    
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score
 
 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 


def find_index(list_of_lists, target):
    for index, sublist in enumerate(list_of_lists):
        if sublist[0] == target:
            return index
    return -1  # If target is not found in the list_of_lists

def f_score(bound_pred, ground_truth, overlap, file_name,bg_class=["background"]):

    #p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)

    _, p_start, p_end = bound_pred[find_index(bound_pred,file_name)]

    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    
    #print('p_startp_startp_startp_startp_start',p_start)
    #print('p_endp_endp_endp_endp_endp_endp_end',p_end)
    #print( "y_starty_starty_starty_start",y_start)
    #print( "y_endy_endy_endy_endy_end",y_end)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_start)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = intersection / union # if union > 0 else 0  # To avoid division by zero
        # Get the best scoring segment

        idx = np.array(IoU).argmax()

        #print('IoUIoUIoUIoUIoUIoUIoU',IoU, type(IoU))
        #print('overlapoverlapoverlapoverlap',overlap)
 
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)




 
 
def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.get_cmap('seismic')
    # color_map =
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=20)
 
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def segment_bars_with_confidence(save_path, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('seismic')
 
    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0)
    fig = plt.figure(figsize=(15, num_pics * 1.5))
    interval = 1 / (num_pics+1)

    for i, label in enumerate(labels):
        i = i + 1
        ax1 = fig.add_axes([0, 1-i*interval, 1, interval])
        ax1.imshow([label], **barprops)
 
    #ax4 = fig.add_axes([0, interval, 1, interval])
    #ax4.set_xlim(0, len(labels[0]))
    #ax4.set_ylim(0, 1)
    #ax4.plot(range(len(labels[0])), labels[0])
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def func_eval(dataset,bound_pred, file_list, ground_truth_path, mapping_file):

    list_of_videos = read_file(file_list).split('\n')[:-1]
 
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
 
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0

    h=0
    for vid in list_of_videos:
 
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]


        #recog_file = recog_path + vid.split('.')[0]
        #print(recog_file)
        #recog_content = read_file(recog_file).split('\n')[1].split()
 
        '''
        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)
        '''


        file_name=vid.split('.')[0]


        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(bound_pred, gt_content,overlap[s],file_name)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    #acc = 100 * float(correct) / total
    #edit = (1.0 * edit) / len(list_of_videos)
    #print("Acc: %.4f" % (acc))
    #print('Edit: %.4f' % (edit))


    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
#         print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1
 
    acc,edit=0,0
    return acc, edit, f1s

def eval_unsup():

  
    if args.dataset=='50salads':
        interval_between_boundaries= 500  #500
    
    if args.dataset=='gtea':
        interval_between_boundaries= 70  #70

    if args.dataset=='breakfast':
        interval_between_boundaries= 300  #100


    seed = 19980125 # my birthday, :)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
    cnt_split_dict = {
        '50salads':5,
        'gtea':4,
        'breakfast':4
    }

    
    model_type='fusion'

    vid_list_file = args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = args.dataset+"/features/"
    ground_truth_path = args.dataset+"/groundTruth/"
    mapping_file = args.dataset+"/mapping.txt"
    recog_path = args.dataset+"/results/split"+ args.split+ "/" +model_type+"/" 
    file_list_test = args.dataset+"/splits/test.split"+args.split+".bundle"
    file_list_train = args.dataset+"/splits/train.split"+args.split+".bundle"


    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    index2label = dict()
    for k,v in actions_dict.items():
        index2label[v] = k
 
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, ground_truth_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    batch_gen_tst.reset()
    import time

    bound_pred=[]

    while batch_gen_tst.has_next():

        batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
        vid = vids[0]
        features = np.load(features_path + vid.split('.')[0] + '.npy')
        features = features[:, ::sample_rate]

        input_x = torch.tensor(features, dtype=torch.float)
        input_x.unsqueeze_(0)
        input_x = input_x.to(device)
        print(vid)

        ps, st, ed = find_correct_boundary(input_x,  use_conv=use_conv,   interval_between_boundaries=interval_between_boundaries)

        bound_pred.append([vid.split('.')[0],st, ed])

    acc_all = 0.
    edit_all = 0.
    f1s_all = [0.,0.,0.]

    split = args.split
    acc_all, edit_all, f1s_all = func_eval(args.dataset, bound_pred, file_list_test, ground_truth_path, mapping_file)
    
    #print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
    print("F1@10 =",round(f1s_all[0],1))
    








def cosine_similarity_seq (seq1,seq2):

    dot_product = np.sum(seq1 * seq2)

    magnitude1 = np.sqrt(np.sum(seq1 ** 2))
    magnitude2 = np.sqrt(np.sum(seq2 ** 2))

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


def similarity(features, sim_wind):

    i=0

    subsequences=[]

    while i <= (features.size(1)-(sim_wind)):
        
        # align together in the same row
        one_seq = features[:,i:i+sim_wind]

        subsequences.append(one_seq.detach().cpu().numpy())

        i+=sim_wind

    # Extract subsequences
    #subsequences = [features[ :, i:i+sim_wind] for i in range(num_subsequences)]

    #print(len(subsequences))

    # Reshape subsequences for clustering
    X = np.array(subsequences).reshape(len(subsequences), -1)

    #print(X.shape)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,  init=init, n_init=n_init)
    kmeans.fit(X)

    # Assign labels to subsequences

    cluster_labels_sequence = kmeans.labels_



    cosine_similarity_indv=[]

    i=0

    while i <= (features.size(1)-2):

        subsequence1 = features[:,i]
        subsequence2 = features[:,i+1]
        distance = cosine_similarity_seq(subsequence1.detach().cpu().numpy(), subsequence2.detach().cpu().numpy())
        cosine_similarity_indv.append(distance)

        i+=1
    cosine_similarity_indv=cosine_similarity_indv/max(cosine_similarity_indv)
    cosine_similarity_indv=np.array([round(num, 2) for num in cosine_similarity_indv])


    dtw_similarity_indv=[]

    i=0

    while i <= (features.size(1)-2):

        subsequence1 = features[:,i]
        subsequence2 = features[:,i+1]
        distance, _ = fastdtw(subsequence1.detach().cpu().numpy(), subsequence2.detach().cpu().numpy())
        dtw_similarity_indv.append(distance)

        i+=1

    dtw_similarity_indv=dtw_similarity_indv/max(dtw_similarity_indv)
    dtw_similarity_indv=np.array([round(num, 2) for num in dtw_similarity_indv])

    

    return cluster_labels_sequence, cosine_similarity_indv, dtw_similarity_indv





def find_largest_sequences_0_and_1(arr):

    max_zeros_start = 0
    max_zeros_end = 0

    max_ones_start = 0
    max_ones_end = 0

    zero_count = 0
    one_count = 0

    for i in range(len(arr)):
        if arr[i] == 0:
            one_count = 0
            zero_count += 1
            if zero_count > max_zeros_end - max_zeros_start:
                max_zeros_start = i - zero_count + 1
                max_zeros_end = i
        else:
            zero_count = 0
            one_count += 1
            if one_count > max_ones_end - max_ones_start:
                max_ones_start = i - one_count + 1
                max_ones_end = i

    list_st_ed=[max_ones_start, max_ones_end, max_zeros_start, max_zeros_end]
    list_st_ed_sorted=sorted(list_st_ed)

    return  list_st_ed_sorted





def values_less_than_mean(arr):
    mean_value = np.mean(arr)  # Calculate the mean of the array
    # Find indices of values less than the mean
    indices_less_than_mean = np.where(arr < mean_value)[0]
    # Get the values less than the mean
    values_less_than_mean = arr[indices_less_than_mean]
    return indices_less_than_mean, values_less_than_mean



def values_greater_than_mean(arr):
    mean_value = np.mean(arr)  # Calculate the mean of the array
    # Find indices of values greater than the mean
    indices_greater_than_mean = np.where(arr > mean_value)[0]
    # Get the values greater than the mean
    values_greater_than_mean = arr[indices_greater_than_mean]
    return indices_greater_than_mean, values_greater_than_mean



def remove_close_values(arr, threshold):

    filtered_arr = [arr[0]]
    prev_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] - prev_value > threshold:
            filtered_arr.append(arr[i])
            prev_value = arr[i]
    

    return np.array(filtered_arr)



def merge_sorted_arrays(arr1, arr2, arr3):
    merged = []
    i = j = k = 0

    while i < len(arr1) and j < len(arr2) and k < len(arr3):
        if arr1[i] <= arr2[j] and arr1[i] <= arr3[k]:
            merged.append(arr1[i])
            i += 1
        elif arr2[j] <= arr1[i] and arr2[j] <= arr3[k]:
            merged.append(arr2[j])
            j += 1
        else:
            merged.append(arr3[k])
            k += 1

    while i < len(arr1):
        merged.append(arr1[i])
        i += 1
    while j < len(arr2):
        merged.append(arr2[j])
        j += 1
    while k < len(arr3):
        merged.append(arr3[k])
        k += 1

    return merged



def calculate_mean_subsequences(arr, threshold):
    means = []
    subsequence = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] - min(subsequence) <= threshold:
            subsequence.append(arr[i])
        else:
            if len(subsequence) > 1:
                means.append(np.mean(subsequence))
            else:
                means.append(subsequence[0])
            subsequence = [arr[i]]
    if len(subsequence) > 1:
        means.append(np.mean(subsequence))
    else:
        means.append(subsequence[0])
    return means



def find_correct_boundary(input_x, use_conv, interval_between_boundaries):
        
        con1d_layer=nn.Conv1d(in_channels=2048, out_channels=feat_dim, kernel_size=1)
        con1d_layer=con1d_layer.to(device)

        #print('interval_between_boundaries',interval_between_boundaries)

        st_bound=[]
        ed_bound=[]

        st_bound.append(0)

        features= torch.squeeze(input_x)

        if use_conv==True:

            features = con1d_layer(features) 

        full_Seq_clusters , cosine_similarity_indv , dtw_similarity_indv= similarity ( features  ,sim_wind= sim_wind)

        cluster_labels=torch.zeros(features.size(1))
        
        h=0

        for j in range (0,len(full_Seq_clusters)):
            cluster_labels[h:h+sim_wind]=full_Seq_clusters[j]
            h+=sim_wind
        
        #p_s= cluster_labels

        boundarie_clusters=[]

        boundarie_clusters.append(0)

        for j in range (0,len(cluster_labels)-1):

            if cluster_labels[j]!=cluster_labels[j+1]:

                boundarie_clusters.append(j+1)
        
        boundarie_clusters.append(len(cluster_labels)-1)




        boundarie_cosine, _ = np.sort(values_less_than_mean(cosine_similarity_indv)) # length_video / num_actions

        boundarie_cosine=np.insert(boundarie_cosine, 0, 0)

        boundarie_cosine = np.append(boundarie_cosine, features.size(1)-1) 

        
        



        boundarie_dtw, boundarie_dtw_val= np.sort(values_greater_than_mean(dtw_similarity_indv))

        boundarie_dtw=np.insert(boundarie_dtw, 0, 0)

        boundarie_dtw = np.append(boundarie_dtw, features.size(1)-1) 

        


        if args.unsup_thresh_auto:

            differences_clustering = [boundarie_clusters[i+1] - boundarie_clusters[i] for i in range(len(boundarie_clusters) - 1)]
            differences_cosine = [boundarie_cosine[i+1] - boundarie_cosine[i] for i in range(len(boundarie_cosine) - 1)]
            differences_dtw = [boundarie_dtw[i+1] - boundarie_dtw[i] for i in range(len(boundarie_dtw) - 1)]

            interval_between_boundaries=max(int(np.max(np.diff(differences_clustering))), int(np.max(np.diff(differences_cosine))) , int(np.max(np.diff(differences_dtw))))
        

        boundarie_clusters = remove_close_values(boundarie_clusters,  interval_between_boundaries)
        boundarie_cosine = remove_close_values(boundarie_cosine,  interval_between_boundaries)
        boundarie_dtw = remove_close_values(boundarie_dtw,  interval_between_boundaries)



        merged_boundaries= np.array(merge_sorted_arrays(boundarie_clusters, boundarie_cosine, boundarie_dtw)).astype(int)

        final_boundaries=np.array(calculate_mean_subsequences(merged_boundaries, interval_between_boundaries)).astype(int)

        final_labels=torch.zeros(features.size(1))


        
        for j in range (0,len(final_boundaries)-1):

            final_labels[int(final_boundaries[j]):int(final_boundaries[j+1])]=j

        final_labels[-1]=j

        p_s=final_labels

        #print('cluster',boundarie_clusters)
        #print('cos',boundarie_cosine)
        #print('dtw',boundarie_dtw)
        #print('merged_boundaries',merged_boundaries)
        #print('final_boundaries',final_boundaries)


        #time.sleep(1000)
       

        #if smth_bool==True:
        #    p_s = smoothing( p_s ,smooth_window=smooth_window)




        k=0

        while k < len(p_s)-1:

            if p_s[k+1]!=p_s[k] :

                ed_bound.append(k)
                st_bound.append(k)

            k+=1 
                

        if len(st_bound)==1:
            ed_bound.append(len(p_s)-1)
        else: 
            if ed_bound[-1]==st_bound[-1]:
                st_bound.pop()



        


        return p_s,st_bound,ed_bound




