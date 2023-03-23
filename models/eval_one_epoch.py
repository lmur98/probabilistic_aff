from statistics import mean
import torch
import numpy as np
from models.baseline import data_to_GPU, object_detection_loss
from common.metrics import compute_epoch_loss, update_stats_affordances, init_loss_stats, init_stats, filter_detections, group_detections, compute_mean_Fb
from models.train_one_epoch import compute_all_losses

def evaluate_one_epoch(model, data_loader, device, epoch, writer): 
    model.eval()
    aff_labels = {0: 'background', 1: 'contain', 2: 'cut', 3: 'display', 4: 'engine', 5: 'grasp', 6: 'hit', 7: 'pound', 8: 'support', 9:'w-grasp'}
    aff_stads = init_stats(aff_labels)
    no_filt_aff_stads = init_stats(aff_labels)
    with torch.no_grad():
        for batch, data in enumerate(data_loader, 0):
            images_cuda, targets_cuda = data_to_GPU(data, device)
            y_pred = model(images_cuda, targets_cuda)
            for i in range(len(y_pred)):
                obs_masks, obs_bboxes, obs_labels = y_pred[0][i]['masks'].cpu().numpy(), y_pred[0][i]['boxes'].cpu().numpy(), y_pred[0][i]['labels'].cpu().numpy()
                gt_mask, gt_boxes, gt_labels = targets_cuda[i]['aff_map'].cpu().numpy(), targets_cuda[i]['boxes'].cpu().numpy(), targets_cuda[i]['labels'].cpu().numpy()
                filter_pred_masks, filter_labels = filter_detections(gt_mask, gt_boxes, gt_labels,
                                                                     obs_masks, obs_bboxes, obs_labels)

                filter_pred_masks = group_detections(filter_pred_masks, filter_labels)
                if obs_masks.shape[0] == 0: #For safety, we might do not have any observation
                    obs_masks = None
                no_filt_pred_masks = group_detections(obs_masks, obs_labels)

                filter_Fb_weighted_score = update_stats_affordances(aff_stads, gt_mask, filter_pred_masks)
                no_filt_Fb_weighted_score = update_stats_affordances(no_filt_aff_stads, gt_mask, no_filt_pred_masks)

            if (batch % 200) == 0:
                print('---VALIDATION----, epoch', epoch,'batch',batch)

        filter_mean_Fb, no_filter_mean_Fb = compute_mean_Fb(no_filt_Fb_weighted_score, filter_Fb_weighted_score, aff_labels)
        writer.add_scalar('Epoch/F_b_metric', filter_mean_Fb, epoch)
        writer.add_scalars('Epoch/F_b_per_ class', {'Contain': np.mean(filter_Fb_weighted_score[1]['q']), 
                                                    'Cut': np.mean(filter_Fb_weighted_score[2]['q']), 
                                                    'Display': np.mean(filter_Fb_weighted_score[3]['q']),
                                                    'Engine': np.mean(filter_Fb_weighted_score[4]['q']), 
                                                    'Grasp': np.mean(filter_Fb_weighted_score[5]['q']), 
                                                    'Hit': np.mean(filter_Fb_weighted_score[6]['q']),
                                                    'Pound': np.mean(filter_Fb_weighted_score[7]['q']), 
                                                    'Support': np.mean(filter_Fb_weighted_score[8]['q']), 
                                                    'W-Grasp': np.mean(filter_Fb_weighted_score[9]['q'])}, epoch)

        writer.add_scalar('Epoch/No_filter_F_b_metric', no_filter_mean_Fb, epoch)
        writer.add_scalars('Epoch/No_filter_F_b_per_ class', {'Contain': np.mean(no_filt_Fb_weighted_score[1]['q']), 
                                                    'Cut': np.mean(no_filt_Fb_weighted_score[2]['q']), 
                                                    'Display': np.mean(no_filt_Fb_weighted_score[3]['q']),
                                                    'Engine': np.mean(no_filt_Fb_weighted_score[4]['q']), 
                                                    'Grasp': np.mean(no_filt_Fb_weighted_score[5]['q']), 
                                                    'Hit': np.mean(no_filt_Fb_weighted_score[6]['q']),
                                                    'Pound': np.mean(no_filt_Fb_weighted_score[7]['q']), 
                                                    'Support': np.mean(no_filt_Fb_weighted_score[8]['q']), 
                                                    'W-Grasp': np.mean(no_filt_Fb_weighted_score[9]['q'])}, epoch)

        
