import numpy as np
import time
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import IIAff_dataset, my_collate
from models.bsas import cluster_vectors, compute_variance, compute_variance, compute_final_prediction, compute_variance_label, compute_binary_variance
from models.baseline import data_to_GPU, baseline
from common.metrics_uncertainty import uncertainty_metrics, group_bayesian_detections, filter_bayesian_detections, one_hot_encoding, PDQ_score, get_probs_for_hotmap
from common.metrics import init_stats, show_all_bboxes, show_all_masks, show_variance, filter_detections, update_stats_affordances, compute_mean_Fb, filter_detections_v2, compute_mean_Fb_v2, remap_to_show

#A. Training on the GPU and tensorboard
gpu_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
aff_labels = {0: 'background', 1: 'contain', 2: 'cut', 3: 'display', 4: 'engine', 5: 'grasp', 6: 'hit', 7: 'pound', 8: 'support', 9:'w-grasp'}
    
#B. Read the dataset and create the dataloader
general_dir = '/home/lmur/documents/iit_aff_DATASET'
test_data = IIAff_dataset(dataset_path = general_dir, mode = 'show') #testing
length = test_data.__len__()
print(length)
#subset_list = list(range(1, len(test_data), 2))
#test_data = torch.utils.data.Subset(test_data, subset_list)

test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False, collate_fn = my_collate)
print('The number of batches in the train datast is: testing:', len(test_dataloader))

#C. Define model, move to device and set dropout to training in eval mode!
forward_passes = [8]
chekpoint = torch.load('/home/lmur/documents/saved_models_05/30_ResneXT_101_dropout_0_5_ENCODER_FC')
model = baseline()
model.load_state_dict(chekpoint['model_state_dict'])
model.eval()
#We set the dropout layers active during inference!
for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
        print('We found a dropout layer', m)
        m.train()

model.to(gpu_device) #Move the model to the GPU

for n in range(len(forward_passes)):
    n_forward_passes = forward_passes[n]
    compute_PDQ_score = PDQ_score()
    unc_metrics = uncertainty_metrics(L_bins = 9, n_classes = 10)
    filter_aff_stads = init_stats(aff_labels)
    no_filter_aff_stads = init_stats(aff_labels)
    all_BS, all_ECE_metrics, all_MCE, all_reliability, all_AUSE = [], [], [], [], []
    tot_PDQ, tot_PDQ_s, tot_PDQ_l, tot_PDQ_fg, tot_PDQ_bg, tot_PDQ_TP, tot_PDQ_FP, tot_PDQ_FN = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader, 0):
            images_cuda, targets_cuda = data_to_GPU(data, gpu_device)
            print('another sample')
            masks_sample, mask_sample_softmax, labels_sample, bboxes_sample = [], [], [], []
            for n_pass in range(n_forward_passes):
                print(targets_cuda[0]['boxes'].shape, 'booxes')
                y_pred = model(images_cuda, targets_cuda)[0]
                gt_mask, gt_boxes, gt_labels = targets_cuda[0]['aff_map'].cpu().numpy(), targets_cuda[0]['boxes'].cpu().numpy(), targets_cuda[0]['labels'].cpu().numpy()
                gt_all_masks = targets_cuda[0]['masks'].cpu().numpy()
                if y_pred[0]['boxes'].shape[0] > 0: #We avoid empty predictions
                    obs_bboxes, obs_labels_probs = y_pred[0]['boxes'].cpu().numpy(), y_pred[0]['labels_probs']
                    obs_masks, obs_masks_sigmoid = y_pred[0]['masks'].cpu().numpy(), y_pred[0]['masks_sigmoid'].cpu().numpy()
                    filter_out = filter_detections_v2(gt_mask, gt_boxes, gt_labels, obs_masks, obs_bboxes, obs_labels_probs, obs_masks_sigmoid)
                    filter_masks, filter_sigmoid, filter_boxes, filter_label_probs = filter_out
                    if filter_boxes is not None:
                        bboxes_sample.append(filter_boxes)
                        masks_sample.append(filter_masks)
                        mask_sample_softmax.append(filter_sigmoid)
                        labels_sample.append(filter_label_probs)
            if len(bboxes_sample) > 0:
                masks_sample = np.concatenate(np.asarray(masks_sample, dtype=object), axis = 0)
                mask_sample_softmax = np.concatenate(np.asarray(mask_sample_softmax, dtype=object), axis = 0)
                labels_sample = np.concatenate(np.asarray(labels_sample, dtype=object), axis = 0)
                bboxes_sample = np.concatenate(np.asarray(bboxes_sample, dtype=object), axis = 0)
                print(masks_sample.shape, mask_sample_softmax.shape, labels_sample.shape, bboxes_sample.shape, 'sgaoes after the concatenation')
                #1. We group the different detections in clusters 
                #print(masks_sample.shape, mask_sample_softmax.shape, labels_sample.shape, bboxes_sample.shape)
                detections_probs, detections_bbox, detections_label, detections_mask = cluster_vectors(masks_sample, 
                                                                                                       mask_sample_softmax, 
                                                                                                       labels_sample, 
                                                                                                       bboxes_sample)
                print('toooo compute the variance')
                print(detections_probs[0].shape, detections_bbox[0].shape, detections_label[0].shape, detections_mask[0].shape)
                detections_dict = compute_binary_variance(detections_mask)
                #detections_dict = compute_variance(detections_probs) 
                print('variance computed')
                detections_labels_dict = compute_variance_label(detections_label)
                masked_map = np.zeros((len(detections_mask), detections_mask[0].shape[1], detections_mask[0].shape[2]))
                for d in range(len(detections_mask)):
                    if detections_mask[d].shape[0] > 5:
                        masked_map[d, :, :] = np.mean(detections_mask[d], axis = 0)
                        masked_map[d, :, :] = np.where(masked_map[d, :, :] > 0.5, detections_labels_dict[d]['trace'], 0)
                #show_all_bboxes(bboxes_sample, data[2][0], detections_bbox)
                #show_all_masks(masks_sample, data[2][0], detections_probs)
                show_variance(data[2][0], detections_dict, detections_bbox, masked_map)

                #2. We extract the mean of the detections to get the observations
                obs_bboxes, filter_labels, filter_probs, filter_label_probs = compute_final_prediction(detections_probs, detections_bbox, detections_label)      
                filter_pred_mask, index_pred = group_bayesian_detections(filter_probs, filter_labels)
                
                if filter_pred_mask is not None:
                    y_pred_init = np.zeros((10, filter_pred_mask.shape[0], filter_pred_mask.shape[1]))
                    y_pred_init[0, :, :] = 1.0 #All is predicted as background
                    #y_pred_init[1:, :, :] = (1 -  0.999999) / 9
                    y_pred = y_pred_init
                    for n in range(filter_probs.shape[0]):
                        y_pred = np.where(index_pred == n, filter_probs[n, :, :, :], y_pred)
                    y_pred /= np.sum(y_pred, axis = 0) #WE HAVE TO NORMALIZE THE PROBABILITY
                    gt_onehot = one_hot_encoding(y_pred_init, gt_labels, gt_mask)
                    hotmap = get_probs_for_hotmap(detections_probs, filter_labels)
                    print(hotmap[0].shape, 'hotmap shape', np.unique(hotmap[0]), 'unique hotmap')
                else:
                    y_pred = np.zeros((10, gt_mask.shape[0], gt_mask.shape[1]))
                    y_pred[0, :, :] = 1.0 #All is predicted as background
                    gt_onehot = one_hot_encoding(y_pred, gt_labels, gt_mask)
                    hotmap = None
                    filter_label_probs = None

            else: #For safety, sometimes there is not any detection:(
                gt_mask = targets_cuda[0]['aff_map'].cpu().numpy()
                filter_pred_mask = np.zeros((gt_mask.shape[0], gt_mask.shape[1]))
                y_pred = np.zeros((10, gt_mask.shape[0], gt_mask.shape[1])) 
                y_pred[0, :, :] = 1.0 #All is predicted as background
                gt_onehot = one_hot_encoding(y_pred, gt_labels, gt_mask)
                hotmap = None
                filter_label_probs = None

            #3. Compute the final metrics with the averaged observations as in the deterministic evaluation
            y_pred_tensor = torch.unsqueeze(torch.from_numpy(y_pred.astype(np.float32)), dim = 0)
            gt_mask_tensor = torch.unsqueeze(torch.from_numpy(gt_mask.astype(np.int64)), dim = 0)
            
            PDQ_metric = compute_PDQ_score.compute_PQD_metric(gt_all_masks, gt_labels, hotmap, filter_label_probs)
            tot_PDQ += PDQ_metric['overall']
            tot_PDQ_s += PDQ_metric['spatial'] 
            tot_PDQ_l += PDQ_metric['label']
            tot_PDQ_bg += PDQ_metric['bg']
            tot_PDQ_fg += PDQ_metric['fg']
            tot_PDQ_TP += PDQ_metric['TP']
            tot_PDQ_FP += PDQ_metric['FP']
            tot_PDQ_FN += PDQ_metric['FN']
            BS_metrics = unc_metrics.AUSE_metric(y_pred, gt_onehot)
            ECE_metrics = unc_metrics.ECE_metric_v2(y_pred, gt_mask)
            all_ECE_metrics.append(ECE_metrics['ECE'])
            all_BS.append(BS_metrics['bs'])
            all_MCE.append(ECE_metrics['MCE'])
            all_reliability.append(ECE_metrics['reliability'])
            all_AUSE.append(BS_metrics['ause'])
            filter_Fb_weighted_score = update_stats_affordances(filter_aff_stads, gt_mask, filter_pred_mask)   
            #remap_to_show(filter_pred_mask, gt_mask, data[2][0])
        print('------------------',n_forward_passes, 'MODELS ----------------')
        compute_mean_Fb_v2(filter_Fb_weighted_score, aff_labels)  
        print('The Brier Score is', format(sum(all_BS) / len(all_BS), ".3f"), 'and AUSE error', format(sum(all_AUSE) / len(all_AUSE), ".3f"))
        print('The ECE is ', format(sum(all_ECE_metrics) / len(all_ECE_metrics), ".8f"), 'and MCE', format(sum(all_MCE) / len(all_MCE), ".3f"))
        print('And the mean reliability diagram is', np.mean(np.asarray(all_reliability), axis = 0))   
        print('PDQ score:', 100 * tot_PDQ / (tot_PDQ_TP + tot_PDQ_FP + tot_PDQ_FN), 'overal PDQ: ', 100 * tot_PDQ / tot_PDQ_TP)
        print('PDQ spatial is', 100 * tot_PDQ_s / tot_PDQ_TP, 'the label is', 100 * tot_PDQ_l / tot_PDQ_TP, 'the FG', 100 * tot_PDQ_fg / tot_PDQ_TP, 'the BG', 100 * tot_PDQ_bg / tot_PDQ_TP)
        print()

    