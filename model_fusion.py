import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim

 
import eval
import copy
import numpy as np 
import math  
from model_32 import Encoder32
from model_64 import Encoder64
from model_128 import Encoder128
from model_256 import Encoder256

from eval import segment_bars_with_confidence
from collections import Counter
import time
from arguments import  args

from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from sklearn.cluster import KMeans
 
import time 
import sys


device = torch.device("cuda:"+args.gpu_num if torch.cuda.is_available() else "cpu")


np.set_printoptions(threshold=sys.maxsize)

random_state=20
init='k-means++'
n_init=100

#window_boundary=16 
#sim_wind=4
feat_bound_corr=64
#smooth_window=80



#print(args)



if args.dataset == "50salads":
    num_classes=19
# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    num_classes=11

if args.dataset == 'breakfast':
    num_classes=48


path_32= args.dataset+"/checkpoints/split"+args.split+ "/32/epoch-"+  str(args.checkpoint32)   +".model" 
path_64=args.dataset+"/checkpoints/split"+args.split+ "/64/epoch-"+  str(args.checkpoint64)   +".model" 
path_128=args.dataset+"/checkpoints/split"+args.split+ "/128/epoch-"+  str(args.checkpoint128)   +".model" 
path_256=args.dataset+"/checkpoints/split"+args.split+ "/256/epoch-"+  str(args.checkpoint256)   +".model" 


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

def majority_prediction_voting(predictions):
   
    num_frames = predictions.shape[3]
    final_predictions = np.zeros((1,1, num_classes, num_frames))
    for frame_idx in range(num_frames):
        frame_predictions = predictions[:, :, :, frame_idx]
        max_indices = np.argmax(frame_predictions, axis=2)
        max_indices = list(max_indices)
#        index_counts = Counter(max_indices)
        matching_indices = []
        for i in range(len(max_indices)):
             for j in range(i + 1, len(max_indices)):
                 if max_indices[i] == max_indices[j]:
                    matching_indices.extend([i, j])
        matching_indices = list(set(matching_indices))
        if len( matching_indices)>=2:
           best_prediction_index = max(matching_indices, key=lambda index: np.max(predictions[index, :, :, frame_idx]))
#           final_predictions = predictions[best_prediction_index,:,:,frame_idx]
           final_predictions[0, 0, :, frame_idx] = predictions[best_prediction_index, 0, :, frame_idx]
           
        else:
           final_predictions[0, 0, :, frame_idx] = predictions[3, 0, :, frame_idx]

    return final_predictions      




class TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        x, mask = x.to(device), mask.to(device)
        x = x.float()
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        outputs = out.unsqueeze(0)
  
        return outputs[-1]   
    
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :] 
        
class VideoSegModel(nn.Module):
    def __init__(self,  r1, r2,  num_classes, channel_masking_rate):
        super(VideoSegModel, self).__init__()

        self.model32 = Encoder32( 10, r1, r2, 32, 2048, num_classes, channel_masking_rate)
        self.model32.load_state_dict(torch.load(path_32))
        self.model32.to(device)
        self.model32.eval()

        self.model64 = Encoder64(10, r1, r2, 64, 2048, num_classes, channel_masking_rate)
        self.model64.load_state_dict(torch.load(path_64))
        self.model64.to(device)
        self.model64.eval()

        self.model128 = Encoder128(10, r1, r2, 128, 2048, num_classes, channel_masking_rate)
        self.model128.load_state_dict(torch.load(path_128))
        self.model128.to(device)
        self.model128.eval()

        self.model256 = Encoder256(10, r1, r2, 256, 2048, num_classes, channel_masking_rate)
        self.model256.load_state_dict(torch.load(path_256))
        self.model256.to(device)
        self.model256.eval()

        self.TCNBlock1 = TCN(5, 64, num_classes, num_classes)
        self.TCNBlock2 = TCN(10, 64, num_classes, num_classes)
        self.TCNBlock3 = TCN(15, 64, num_classes, num_classes)
        self.TCNBlock4 = TCN(20, 64, num_classes, num_classes)

        self.DilResB1=DilatedResidualLayer(40, num_classes, num_classes)
        self.DilResB2=DilatedResidualLayer(40, num_classes, num_classes)
        self.DilResB3=DilatedResidualLayer(40, num_classes, num_classes)
        self.DilResB4=DilatedResidualLayer(40, num_classes, num_classes)

       
    def forward(self, x, mask):

        predictions =[]
        out0, feature0 =  self.model32(x,mask)
        out1, feature1 =  self.model64(x,mask)
        out2, feature2 = self.model128(x,mask)
        out3, feature3 = self.model256(x,mask)

        all_predictions = np.concatenate([out0.cpu().detach().numpy(), out1.cpu().detach().numpy(), out2.cpu().detach().numpy(), out3.cpu().detach().numpy()], axis=0)
        mid_result = majority_prediction_voting(all_predictions)    

        inp = torch.from_numpy(mid_result)
        inp = inp.squeeze(dim=0)
            
        if args.action=='train' or args.action=='load' or args.bound_in_pred==True:

            inp_reshaped=inp.permute(0,2,1)

            lab=pred_to_labels(inp_reshaped)
            
            
            fet=torch.cat((x,feature1,feature2,feature2),1)

            _, inp = correct_boundary(lab ,x, window_boundary=args.bound_wind ,sim_wind=args.sim_wind, feat_bound_corr=feat_bound_corr, in_chn=2048,score_pred=inp)
            

            #_, inp0 = correct_boundary(lab ,feature0, window_boundary=args.bound_wind ,sim_wind=args.sim_wind, feat_bound_corr=feat_bound_corr, in_chn=32,score_pred=inp)
            
            #_, inp1 = correct_boundary(lab ,feature1, window_boundary=args.bound_wind ,sim_wind=args.sim_wind, feat_bound_corr=feat_bound_corr, in_chn=64,score_pred=inp)
            
            #_, inp2 = correct_boundary(lab ,feature2, window_boundary=args.bound_wind ,sim_wind=args.sim_wind, feat_bound_corr=feat_bound_corr, in_chn=128,score_pred=inp)
            
            #_, inp3 = correct_boundary(lab ,feature3, window_boundary=args.bound_wind ,sim_wind=args.sim_wind, feat_bound_corr=feat_bound_corr, in_chn=256,score_pred=inp)

        

        B_1 = self.TCNBlock1(inp,mask)
        B_1 = self.DilResB1(B_1,mask)

        B_2 = self.TCNBlock2(B_1,mask) 
        B_2 = self.DilResB1(B_2,mask)

        B_3 = self.TCNBlock3(B_2,mask)
        B_3 = self.DilResB1(B_3,mask)
        
        B_4 = self.TCNBlock4(B_3,mask)
        B_4 = self.DilResB1(B_4,mask)


        predictions.append(B_1)
        predictions.append(B_2)
        predictions.append(B_3)
        predictions.append(B_4)


        return predictions
        

    
class Trainer:
    
    def __init__(self, r1, r2,  num_classes, channel_masking_rate):
        self.model = VideoSegModel(r1, r2, num_classes, channel_masking_rate).to(device)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR:{}'.format(learning_rate))
        
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                #print('iteration')
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                optimizer.zero_grad()
                ps = self.model(batch_input ,mask)
                
                loss = 0
                for p in ps:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.requires_grad_()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(ps[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            
            
            scheduler.step(epoch_loss)
            batch_gen.reset()
            print("[epoch %d]: epoch loss = %f,   train_acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total),  end="    ")
            #if (epoch + 1) % 10 == 0  and batch_gen_tst is not None:
            self.test(batch_gen_tst, epoch)
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

    def load(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, last_epoch,batch_gen_tst=None):
        self.model.eval()
        self.model.load_state_dict(torch.load(save_dir + "/epoch-" + str(last_epoch) + ".model"))
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        print('LR:{}'.format(learning_rate))
        
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in range(last_epoch, num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                #print('iteration')
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                optimizer.zero_grad()
                ps = self.model(batch_input ,mask)
                
                loss = 0
                for p in ps:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.requires_grad_()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(ps[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            
            
            scheduler.step(epoch_loss)
            batch_gen.reset()
            print("[epoch %d]: epoch loss = %f,   train_acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total), end="    ")
            #if (epoch + 1) % 10 == 0  and batch_gen_tst is not None:
            self.test(batch_gen_tst, epoch)
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
        

    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct = 0
        total = 0
        if_warp = False  # When testing, always false
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
#                p = self.model(batch_input, mask)[-1].unsqueeze(dim=0).to(device)
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

        acc = float(correct) / total
        print("test_acc = %f" % (acc))

        self.model.train()
        batch_gen_tst.reset()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):


        self.model.eval()

        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

            batch_gen_tst.reset()
            import time
            
            time_start = time.time()
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                
                vid = vids[0]
                
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions_encod_tcn = self.model(input_x,torch.ones(input_x.size(), device=device))
                predictions_tcn4=predictions_encod_tcn[3].permute(0,2,1)
                batch_target = batch_target.squeeze()
                list_pred=[]

                pred = pred_to_labels(predictions_tcn4)

                segment_bars_with_confidence(results_dir + '/{}_pred.png'.format(vid),batch_target.tolist(), pred.tolist())

                if args.bound_aft_pred:
                    pred, _ = correct_boundary(pred ,input_x,window_boundary=args.bound_wind ,sim_wind=args.sim_wind, feat_bound_corr=feat_bound_corr,in_chn=2048, score_pred=None)
                    segment_bars_with_confidence(results_dir + '/{}_correct.png'.format(vid),batch_target.tolist(), pred.tolist())
                
                if args.auto_smooth:
                    bound_list=calculat_number_of_boundaries(pred)
                    sm_wd=int(max_distance(bound_list)/10)
                else:
                    sm_wd=args.smth_wind
                    
                pred = smoothing( pred ,smooth_window = sm_wd ) # best 80 salades try more, 6 gtea
                segment_bars_with_confidence(results_dir + '/{}_smooth.png'.format(vid),batch_target.tolist(), pred.tolist())
                
                
                recognition = []
                for i in range(len(pred)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(pred[i].item())]] * sample_rate))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

            time_end = time.time()
        
        eval.main('fusion')





def nearest_even_value_and_divisor(n):

    list_nearest=[n,n+1,n-1]


    for k in list_nearest:
        list_divisors=[]
        for i in range (4,k):
            if k%i==0:
                list_divisors.append(i)
        if len(list_divisors)>0:
            divisor=int(min(list_divisors))
            nearest=k
            break

    return nearest, divisor
     

def find_divisor(n):
    # Function to find a divisor of n within specific conditions

    if n % 2 != 0:
        n += 1


    for d in range(4, n):
        if n % d == 0:
            return d
    for d in range(2, n):
        if n % d == 0:
            return d
    return None  # If no divisor is found (which theoretically shouldn't happen)

def average_distance(boundaries):
    # Ensure there are at least two boundaries to calculate a distance
    if len(boundaries) < 2:
        return 0  # Or raise an exception depending on the use case
    
    # Calculate the distances between consecutive boundaries
    distances = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
    
    # Calculate the average of these distances
    average = sum(distances) / len(distances)
    
    return average


def min_distance(values):
    min_dist = float('inf')  # Initialize with a large value
    for i in range(len(values) - 1):
        dist = values[i+1] - values[i]  # Calculate difference between consecutive values
        if dist < min_dist:
            min_dist = dist
    return min_dist


def max_distance(sorted_values):
    max_dist = float('-inf')  # Initialize with a small value
    for i in range(len(sorted_values) - 1):
        dist = sorted_values[i+1] - sorted_values[i]  # Calculate difference between consecutive values
        if dist > max_dist:
            max_dist = dist
    return max_dist

def calculat_number_of_boundaries(p_s):
    num_bundaries=[]
    for k in range (0,len(p_s)-1):
        if p_s[k+1]!=p_s[k]:
            num_bundaries.append(k+1)
    return num_bundaries    




def pred_to_labels(prediction_score):
    confidence, predicted = torch.max(F.softmax(prediction_score[0], dim=1).data, 1)
    confidence, predicted = confidence.squeeze().squeeze(), predicted.squeeze().squeeze()
    return predicted

def cosine_similarity_seq (seq1,seq2):
    dot_product = np.sum(seq1 * seq2)
    magnitude1 = np.sqrt(np.sum(seq1 ** 2))
    magnitude2 = np.sqrt(np.sum(seq2 ** 2))
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity


def similarity(features, sim_wind, op):

    features.size(1)

    if op=='all':

        #features= torch.permute(features,(1,0))
        pairwise_similarity_indiv = cosine_similarity(torch.permute(features,(1,0)).detach().cpu().numpy())
        n = pairwise_similarity_indiv.shape[0]
        elements_cosin_indiv = []
        for i in range(n - 1):
            elements_cosin_indiv.append(pairwise_similarity_indiv[i, i + 1])
        elements_cosin_indiv=np.array([round(num, 3) for num in elements_cosin_indiv])
        cosine_similarity_sequence=[]

        i=0
        while i <= (features.size(1)-(2*sim_wind)):


            subsequence1 = features[:,i:i+sim_wind]
            subsequence2 = features[:,i+sim_wind:i+(2*sim_wind)]
            distance = cosine_similarity_seq(subsequence1.detach().cpu().numpy(), subsequence2.detach().cpu().numpy())
            cosine_similarity_sequence.append(distance)

            i+=sim_wind
        cosine_similarity_sequence=cosine_similarity_sequence/max(cosine_similarity_sequence)
        cosine_similarity_sequence=np.array([round(num, 2) for num in cosine_similarity_sequence])

        dtw_similarity_sequence=[]

        i=0

        while i <= (features.size(1)-(2*sim_wind)):

            subsequence1 = features[:,i:i+sim_wind]
            subsequence2 = features[:,i+sim_wind:i+(2*sim_wind)]
            distance, _ = fastdtw(subsequence1.detach().cpu().numpy(), subsequence2.detach().cpu().numpy())
            dtw_similarity_sequence.append(distance)

            i+=sim_wind

        dtw_similarity_sequence=dtw_similarity_sequence/max(dtw_similarity_sequence)
        dtw_similarity_sequence=np.array([round(num, 2) for num in dtw_similarity_sequence])

        reshaped_sequence = features.reshape(-1, features.size(0))

        num_clusters = 2

        kmeans = KMeans(n_clusters=num_clusters,random_state=random_state,  init=init, n_init=n_init)
        kmeans.fit(reshaped_sequence.detach().cpu().numpy())

        cluster_labels_indiv = kmeans.labels_
        F, T = features.shape
        i=0
        subsequences=[]

        while i <= (features.size(1)-(sim_wind)):
            # align together in the same row
            one_seq = features[:,i:i+sim_wind]
            subsequences.append(one_seq.detach().cpu().numpy())
            i+=sim_wind
        X = np.array(subsequences).reshape(len(subsequences), -1)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        cluster_labels_sequence = kmeans.labels_
        return elements_cosin_indiv, cosine_similarity_sequence, dtw_similarity_sequence, cluster_labels_indiv, cluster_labels_sequence

    else:


        reshaped_sequence = features.reshape(-1, features.size(0))
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters,random_state=random_state,  init=init, n_init=n_init)
        kmeans.fit(reshaped_sequence.detach().cpu().numpy())
        cluster_labels_indiv = kmeans.labels_

        return cluster_labels_indiv





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




def correct_boundary(p_s ,input_x, window_boundary, sim_wind, feat_bound_corr, in_chn,score_pred=None):

        k=0


        con1d_layer=nn.Conv1d(in_channels=in_chn, out_channels=feat_bound_corr, kernel_size=1)
        con1d_layer=con1d_layer.to(device)

        while k < len(p_s)-1:

            if p_s[k+1]!=p_s[k] and ((k-window_boundary//2)>0):

                features= torch.squeeze(input_x)[:,k-window_boundary//2 : k+window_boundary//2]
                features = con1d_layer(features) 
                if (features.size(1) < (2*sim_wind)):
                    sim_wind=features.size(1)//2

                cosin_indv, cosin_sequence, dtw_sequence, cluster_indiv, cluster_sequence = similarity(features,sim_wind,op='all')
                
                cs_ind_idx=np.argmin(cosin_indv)      # window_boundary-1
                cs_seq_idx=np.argmin(cosin_sequence)  # window_boundary/sim_wind  -1 
                dtw_seq_idx=np.argmax(dtw_sequence)   # window_boundary/sim_wind  -1 
                list_cl_ind=find_largest_sequences_0_and_1(cluster_indiv)   # window_boundary
                list_cl_seq=find_largest_sequences_0_and_1(cluster_sequence) # window_boundary/sim_wind

                bound_cs_st = (cs_seq_idx+1) * sim_wind  #################### because the new action start in the next segment
                bound_cs_ed = (cs_seq_idx+1) * sim_wind + sim_wind  

                bound_dtw_st = (dtw_seq_idx+1) * sim_wind 
                bound_dtw_ed = (dtw_seq_idx+1) * sim_wind  + sim_wind

                bound_cl_st = ((list_cl_seq[2]-1)+1) * sim_wind
                bound_cl_ed = ((list_cl_seq[2]-1)+1) * sim_wind + sim_wind

                boundary_st= min(bound_cs_st, bound_dtw_st, bound_cl_st)
                boundary_ed= max(bound_cs_ed, bound_dtw_ed, bound_cl_ed)

                boundary_st_real=  (k-window_boundary//2)  +boundary_st 
                boundary_ed_real = boundary_st_real + (boundary_ed-boundary_st)

                interval=boundary_ed_real-boundary_st_real

                prev_interval=0

                while  interval>=(2*sim_wind): # and (interval != prev_interval) 

                    features= torch.squeeze(input_x)[:,boundary_st_real : boundary_ed_real]
                    features = con1d_layer(features) 

                    cosin_indv, cosin_sequence, dtw_sequence, cluster_indiv, cluster_sequence = similarity(features,sim_wind,op='all')
                
                    cs_ind_idx=   np.argmin(cosin_indv)      # window_boundary-1
                    cs_seq_idx=   np.argmin(cosin_sequence)  # window_boundary/sim_wind  -1 
                    dtw_seq_idx=  np.argmax(dtw_sequence)   # window_boundary/sim_wind  -1 
                    list_cl_ind=  find_largest_sequences_0_and_1(cluster_indiv)   # window_boundary
                    list_cl_seq=  find_largest_sequences_0_and_1(cluster_sequence) # window_boundary/sim_wind

                    bound_cs_st = (cs_seq_idx+1) * sim_wind
                    bound_cs_ed = (cs_seq_idx+1) * sim_wind + sim_wind

                    bound_dtw_st = (dtw_seq_idx+1) * sim_wind 
                    bound_dtw_ed = (dtw_seq_idx+1) * sim_wind  + sim_wind

                    bound_cl_st = ((list_cl_seq[2]-1)+1) * sim_wind
                    bound_cl_ed = ((list_cl_seq[2]-1)+1) * sim_wind + sim_wind

                    boundary_st+= min(bound_cs_st, bound_dtw_st, bound_cl_st)
                    boundary_ed= boundary_st+ (max(bound_cs_ed, bound_dtw_ed, bound_cl_ed)- min(bound_cs_st, bound_dtw_st, bound_cl_st))


                    boundary_st_real = (k-window_boundary//2) + boundary_st 
                    boundary_ed_real = boundary_st_real + (boundary_ed-boundary_st)

                    prev_interval=interval
                    interval=boundary_ed_real-boundary_st_real

                features= torch.squeeze(input_x)[:,boundary_st_real : boundary_ed_real]
                features = con1d_layer(features) 

                cl_indv_final= similarity(features,sim_wind,op='just_cluster')

                cl_ind_final=find_largest_sequences_0_and_1(cl_indv_final)

                final_boundary_real=boundary_st_real+cl_ind_final[2] # 2 is the start of the second largest sequenc 0 or 1

                np.set_printoptions(threshold=np.inf)

                p_s[k-window_boundary//2 : final_boundary_real]= p_s[k-1]

                p_s[final_boundary_real : k+window_boundary//2]= p_s[k+1].clone()

                if score_pred is not None:

                    for g in range (k-window_boundary//2, final_boundary_real):
                        score_pred[:,:,g]= torch.squeeze(score_pred[:,:,k-1])

                    for g in range (final_boundary_real , (k+window_boundary//2)):
                        if g<score_pred.size(2):
                            score_pred[:,:,g]= torch.squeeze(score_pred[:,:,k+1])

                np.set_printoptions(threshold=np.inf)
                k+=  window_boundary//2

            else:
                k+=1 
        return p_s, score_pred


    
def smoothing(video_predictions_tensor, smooth_window):

    corrected_predictions = []
    lenth=len(video_predictions_tensor)
 
    for i in range(0, lenth-smooth_window, smooth_window):

        window = video_predictions_tensor[i:i+smooth_window]
        most_common_prediction_window = torch.mode(window).values.item()
        next_window_start = i + smooth_window
        next_window_end = min(i + 2 * smooth_window, lenth)
        most_common_prediction_next_window = torch.mode(video_predictions_tensor[next_window_start:next_window_end]).values.item()
        next_window = video_predictions_tensor[next_window_start:next_window_end]

        if next_window_end != lenth-1 : 
            most_common_prediction_next_window = torch.mode(next_window).values.item()
            if most_common_prediction_window == most_common_prediction_next_window:
                video_predictions_tensor[i:i+smooth_window] = most_common_prediction_window
            else:
                for k in range(0,smooth_window):
                    if  window[k] == most_common_prediction_next_window:  
                        if k>1:
                            window[0:k-1] = torch.mode(window[0:k-1]).values.item()
                            video_predictions_tensor[i:i+(k-1)]=torch.mode(window[0:k-1]).values.item()
                            video_predictions_tensor[i+(k-1):i+smooth_window]=most_common_prediction_next_window
                            break
                        else:
                            window[k-1] = torch.mode(window[k-1]).values.item()
                            video_predictions_tensor[i:i+(k-1)]=torch.mode(window[k-1]).values.item()
                            video_predictions_tensor[i+(k-1):i+smooth_window]=most_common_prediction_next_window
                            break

        else:

            video_predictions_tensor[i:i+smooth_window]=most_common_prediction_window
            video_predictions_tensor[next_window_start:next_window_end]=most_common_prediction_next_window

            break

    return video_predictions_tensor

'''
def vot_prediction (list_pred):
    fin_pred = torch.zeros(list_pred[0].size(0))
    lst=[]

    for i in range(0,list_pred[0].size(0)):
        lst=[]
        lst.append(list_pred[0][i])
        lst.append(list_pred[1][i])
        most_common_element=max(set(lst), key = lst.count)
        if len(set(lst))==len(lst):
            fin_pred[i]=list_pred[1][i]
        else:
            fin_pred[i]=most_common_element
    return fin_pred
'''
    




if __name__ == '__main__':
    pass
