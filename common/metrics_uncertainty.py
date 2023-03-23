from typing import final
import numpy as np
import scipy
import scipy.ndimage
from PIL import Image
import torch
import random
import os
from common.metrics import get_iou_bboxes
import cv2
from scipy.stats.mstats import gmean
from scipy.optimize import linear_sum_assignment
from PIL import Image

def filter_bayesian_detections(gt_bboxes, gt_label, pred_bboxes, pred_label, pred_probs):
    # https://github.com/beapc18/AffordanceNet/blob/main/utils/eval_utils.py
    if pred_bboxes.shape[0] > 0:
        filter_labels, filter_probs = [], []
        for obs in range(pred_bboxes.shape[0]):
            max_IoU = 0 #We have to filter the different observations!!
            for box in range(gt_bboxes.shape[0]):
                IoU = get_iou_bboxes(gt_bboxes[box, :], pred_bboxes[obs, :])
                if IoU > max_IoU:
                    max_IoU = IoU
                    max_IoU_gt_label = gt_label[box]
                    max_IoU_pred_label = pred_label[obs]
            if max_IoU > 0.3 and (max_IoU_gt_label == max_IoU_pred_label): #Threshold in the paper!
                filter_labels.append(max_IoU_pred_label)
                filter_probs.append(pred_probs[obs, :, :, :])
        if len(filter_labels) != 0:
            filter_labels = np.asarray(filter_labels)
            filter_probs = np.asarray(filter_probs)
        else:
            filter_labels, filter_probs = None, None
    else:
        filter_labels, filter_probs = None, None

    return filter_labels, filter_probs

def get_probs_for_hotmap(full_probs_map, pred_labels):
    full_binary = []
    for d in range(len(full_probs_map)):
        y_pred = full_probs_map[d]
        l_pred = pred_labels[d]
        y_bin = np.zeros((y_pred.shape[2], y_pred.shape[3]))
        for obs in range(y_pred.shape[0]):
            binary = np.where(((y_pred[obs, l_pred, :, :] > 0.5)), 1, 0)
            y_bin += binary
        y_bin /= y_pred.shape[0]
        full_binary.append(y_bin)
    return full_binary
        
def group_bayesian_detections(pred_probs, pred_labels):
    if pred_probs is not None:
        sorted_pred = np.zeros((pred_probs.shape[0], pred_probs.shape[2], pred_probs.shape[3]))
        for obs in range(pred_probs.shape[0]):
            sorted_pred[obs, :, :] = np.where(pred_probs[obs, pred_labels[obs], :, :] > 0.5, pred_labels[obs], 0)
        print(sorted_pred.shape, 'shape of sorted pred', pred_labels[obs].shape)
        global_pred = np.squeeze(np.max(sorted_pred, axis = 0).astype(int)) #We take the affordance with the maximun label when there is mismatch
        index_pred = np.squeeze(np.argmax(sorted_pred, axis = 0).astype(int)) #We know which detection we use in each pixel
    else:
        global_pred, index_pred = None, None
    return global_pred, index_pred


def one_hot_encoding(y_pred_init, gt_labels, gt_mask):
    gt_onehot = np.zeros_like(y_pred_init)
    gt_onehot[0, :, :] = 1.0
    for l in range(len(gt_labels)):
        ones = np.zeros_like(y_pred_init)
        ones[gt_labels[l], :, :] = 1
        gt_onehot = np.where(gt_mask == gt_labels[l], ones, gt_onehot)
    return gt_onehot


class PDQ_score():
    def __init__(self):
        self.small_number = 1e-14

    def cal_label_qual(self, gt_label_vec, det_label_prob_mat):
        label_qual_mat = det_label_prob_mat[:, gt_label_vec].T.astype(np.float32)     # g x d
        return label_qual_mat

    def cal_fg_loss(self, gt_seg_mat, det_seg_heatmap_mat):
        log_heatmap_mat = np.log(det_seg_heatmap_mat + self.small_number) 
        fg_loss_sum = np.tensordot(gt_seg_mat, log_heatmap_mat, axes=([0, 1], [0, 1]))  # g x d
        return fg_loss_sum

    def cal_bg_loss(self, bg_seg_mat, det_seg_heatmap_mat):
        bg_log_loss_mat = np.log(1 - det_seg_heatmap_mat + self.small_number) * (det_seg_heatmap_mat > 0)
        bg_loss_sum = np.tensordot(bg_seg_mat, bg_log_loss_mat, axes=([0, 1], [0, 1]))  # g x d
        return bg_loss_sum

    def cal_spatial_qual(self, fg_loss_sum, bg_loss_sum, num_fg_pixels_vec):
        total_loss = fg_loss_sum + bg_loss_sum
        loss_per_gt_pixel = total_loss/num_fg_pixels_vec
        spatial_quality = np.exp(loss_per_gt_pixel)
        # Deal with tiny floating point errors or tiny errors caused by _SMALL_VAL that prevent perfect 0 or 1 scores
        spatial_quality[np.isclose(spatial_quality, 0)] = 0
        spatial_quality[np.isclose(spatial_quality, 1)] = 1
        return spatial_quality
    
    def prepare_gt_data(self, gt_masks, gt_labels):
        bg_seg_mat = np.logical_not(gt_masks)
        bg_seg_mat = np.transpose(bg_seg_mat, (1, 2, 0))
        gt_seg_mat = np.transpose(gt_masks, (1, 2, 0)).astype(bool)   # h x w x g
        num_fg_pixels_vec = np.zeros((gt_seg_mat.shape[2], 1))
        for g in range(gt_seg_mat.shape[2]):
            num_fg_pixels_vec[g, 0] = np.count_nonzero(gt_seg_mat[:, :, g])
        gt_label_vec = gt_labels  # g,
        return gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec
    
    def prepare_pred_data(self, hotmap, pred_labels):
        det_seg_heatmap_mat = np.transpose(hotmap, (1, 2, 0))
        return det_seg_heatmap_mat, pred_labels

    def calc_overall_qual(self, label_qual, spatial_qual):
        combined_mat = np.dstack((label_qual, spatial_qual))
        with np.errstate(divide='ignore'):
            overall_qual_mat = gmean(combined_mat, axis=2)
        return overall_qual_mat

    
    def gen_cost_tables(self, gt_masks, gt_labels, hotmap, pred_labels):
        # Initialise cost tables
        n_pairs = max(gt_labels.shape[0], pred_labels.shape[0])
        overall_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
        spatial_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
        label_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
        bg_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
        fg_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)

        # Generate all the matrices needed for calculations
        gt_seg_mat, bg_seg_mat, num_fg_pixels_vec, gt_label_vec = self.prepare_gt_data(gt_masks, gt_labels)
        det_seg_heatmap_mat, det_label_prob_mat = self.prepare_pred_data(hotmap, pred_labels)
        #for i in range(bg_seg_mat.shape[2]):
        #    vis = bg_seg_mat[:, :, i]
        #    im = Image.fromarray(vis)
        #    im.save('/home/lmur/Documents/iit_aff/imgs/bg_mask_'+str(i)+'.jpeg')
        #for i in range(gt_seg_mat.shape[2]):
        #    vis = gt_seg_mat[:, :, i]
        #    im = Image.fromarray(vis)
        #    im.save('/home/lmur/Documents/iit_aff/imgs/fg_mask_'+str(i)+'.jpeg')
        #for i in range(det_seg_heatmap_mat.shape[2]):
        #    vis = (det_seg_heatmap_mat[:, :, i] * 255).astype(np.uint8)
        #    im = Image.fromarray(vis)
        #    im.save('/home/lmur/Documents/iit_aff/imgs/pred_mask_'+str(i)+'.jpeg')
        

        # Calculate spatial and label qualities
        label_qual_mat = self.cal_label_qual(gt_label_vec, det_label_prob_mat)
        fg_loss = self.cal_fg_loss(gt_seg_mat, det_seg_heatmap_mat)
        bg_loss = self.cal_bg_loss(bg_seg_mat, det_seg_heatmap_mat)
        spatial_qual = self.cal_spatial_qual(fg_loss, bg_loss, num_fg_pixels_vec)
        
        # Calculate foreground quality
        fg_loss_per_gt_pixel = fg_loss/num_fg_pixels_vec
        fg_qual = np.exp(fg_loss_per_gt_pixel)
        fg_qual[np.isclose(fg_qual, 0)] = 0
        fg_qual[np.isclose(fg_qual, 1)] = 1

        # Calculate background quality
        bg_loss_per_gt_pixel = bg_loss/num_fg_pixels_vec
        bg_qual = np.exp(bg_loss_per_gt_pixel)
        bg_qual[np.isclose(bg_qual, 0)] = 0
        bg_qual[np.isclose(bg_qual, 1)] = 1
        

        # Generate the overall cost table (1 - overall quality)
        overall_cost_table[:gt_labels.shape[0], :pred_labels.shape[0]] -= self.calc_overall_qual(label_qual_mat,
                                                                                      spatial_qual)
        # Generate the spatial and label cost tables
        spatial_cost_table[:gt_labels.shape[0], :pred_labels.shape[0]] -= spatial_qual
        label_cost_table[:gt_labels.shape[0], :pred_labels.shape[0]] -= label_qual_mat

        # Generate foreground and background cost tables
        fg_cost_table[:gt_labels.shape[0], :pred_labels.shape[0]] -= fg_qual
        bg_cost_table[:gt_labels.shape[0], :pred_labels.shape[0]] -= bg_qual
        return {'overall': overall_cost_table, 'spatial': spatial_cost_table, 'label': label_cost_table, 'fg': fg_cost_table, 'bg': bg_cost_table}
        
    
    def compute_PQD_metric(self, gt_masks, gt_labels, hotmap, pred_labels):
        if hotmap is None or gt_labels.shape[0] == 0:
            return {'overall': 0.0, 'spatial': 0.0, 'label': 0.0, 'fg': 0.0, 'bg': 0.0, 'TP': 0, 'FP': 0,
                'FN': 0, "img_det_evals": [], "img_gt_evals": []}

        else:
            # Record the full evaluation details for every match
            img_det_evals = []
            img_gt_evals = []

            cost_tables = self.gen_cost_tables(gt_masks, gt_labels, hotmap, pred_labels)
            row_idxs, col_idxs = linear_sum_assignment(cost_tables['overall'])
            # Transform the loss tables back into quality tables with values between 0 and 1
            overall_quality_table = 1 - cost_tables['overall']
            spatial_quality_table = 1 - cost_tables['spatial']
            label_quality_table = 1 - cost_tables['label']
            fg_quality_table = 1 - cost_tables['fg']
            bg_quality_table = 1 - cost_tables['bg']

            # Go through all optimal assignments and summarize all pairwise statistics
            # Calculate the number of TPs, FPs, and FNs for the image during the process
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for match_idx, match in enumerate(zip(row_idxs, col_idxs)):
                row_id, col_id = match
                det_eval_dict = {"det_id": int(col_id), "gt_id": int(row_id), "matched": True, "ignore": False,
                         "pPDQ": float(overall_quality_table[row_id, col_id]),
                         "spatial": float(spatial_quality_table[row_id, col_id]),
                         "label": float(label_quality_table[row_id, col_id]),
                         'fg': float(fg_quality_table[row_id, col_id]),
                         'bg': float(bg_quality_table[row_id, col_id]),
                         "correct_class": None}
                gt_eval_dict = det_eval_dict.copy()
                if overall_quality_table[row_id, col_id] > 0:
                    det_eval_dict["correct_class"] = gt_labels[row_id]
                    gt_eval_dict["correct_class"] = gt_labels[row_id]
                    if row_id < gt_labels.shape[0]:
                        true_positives += 1
                    else:
                        overall_quality_table[row_id, col_id] = 0.0
                    img_det_evals.append(det_eval_dict)
                    img_gt_evals.append(gt_eval_dict)
                else:
                    if row_id < gt_labels.shape[0]:
                        gt_eval_dict["correct_class"] = gt_labels[row_id]
                        gt_eval_dict["det_id"] = None
                        gt_eval_dict["matched"] = False
                        false_negatives += 1
                        img_gt_evals.append(gt_eval_dict)

                    if col_id < pred_labels.shape[0]:
                        det_eval_dict["gt_id"] = None
                        det_eval_dict["matched"] = False
                        false_positives += 1
                        img_det_evals.append(det_eval_dict)

            # Calculate the sum of quality at the best matching pairs to calculate total qualities for the image
            tot_overall_img_quality = np.sum(overall_quality_table[row_idxs, col_idxs])

            # Force spatial and label qualities to zero for total calculations as there is no actual association between
            # detections and therefore no TP when this is the case.
            spatial_quality_table[overall_quality_table == 0] = 0.0
            label_quality_table[overall_quality_table == 0] = 0.0
            fg_quality_table[overall_quality_table == 0] = 0.0
            bg_quality_table[overall_quality_table == 0] = 0.0

            # Calculate the sum of spatial and label qualities only for TP samples
            tot_tp_spatial_quality = np.sum(spatial_quality_table[row_idxs, col_idxs])
            tot_tp_label_quality = np.sum(label_quality_table[row_idxs, col_idxs])
            tot_tp_fg_quality = np.sum(fg_quality_table[row_idxs, col_idxs])
            tot_tp_bg_quality = np.sum(bg_quality_table[row_idxs, col_idxs])

            # Sort the evaluation details to match the order of the detections and ground truths
            img_det_eval_idxs = [det_eval_dict["det_id"] for det_eval_dict in img_det_evals]
            img_gt_eval_idxs = [gt_eval_dict["gt_id"] for gt_eval_dict in img_gt_evals]
            img_det_evals = [img_det_evals[idx] for idx in np.argsort(img_det_eval_idxs)]
            img_gt_evals = [img_gt_evals[idx] for idx in np.argsort(img_gt_eval_idxs)]
            #print('overall', 100 * tot_overall_img_quality / (true_positives + false_positives + false_negatives), 'spatial', 100 * tot_tp_spatial_quality / true_positives, 'label', 100 * tot_tp_label_quality / true_positives,
            #'fg', 100 * tot_tp_fg_quality / true_positives, 'bg', 100 * tot_tp_bg_quality / true_positives,
            #'TP', true_positives, 'FP', false_positives, 'FN', false_negatives)

            return {'overall': tot_overall_img_quality, 'spatial': tot_tp_spatial_quality, 'label': tot_tp_label_quality,
            'fg': tot_tp_fg_quality, 'bg': tot_tp_bg_quality,
            'TP': true_positives, 'FP': false_positives, 'FN': false_negatives,
            'img_gt_evals': img_gt_evals, 'img_det_evals': img_det_evals}


def group_bayesian_detections(pred_probs, pred_labels):
    if pred_probs is not None:
        sorted_pred = np.zeros((pred_probs.shape[0], pred_probs.shape[2], pred_probs.shape[3]))
        for obs in range(pred_probs.shape[0]):
            sorted_pred[obs, :, :] = np.where(pred_probs[obs, pred_labels[obs], :, :] > 0.5, pred_labels[obs], 0)
        global_pred = np.squeeze(np.max(sorted_pred, axis = 0).astype(int)) #We take the affordance with the maximun label when there is mismatch
        index_pred = np.squeeze(np.argmax(sorted_pred, axis = 0).astype(int)) #We know which detection we use in each pixel
    else:
        global_pred, index_pred = None, None
    return global_pred, index_pred

def one_hot_encoding(y_pred_init, gt_labels, gt_mask):
    gt_onehot = np.zeros_like(y_pred_init)
    gt_onehot[0, :, :] = 1.0
    for l in range(len(gt_labels)):
        ones = np.zeros_like(y_pred_init)
        ones[gt_labels[l], :, :] = 1
        gt_onehot = np.where(gt_mask == gt_labels[l], ones, gt_onehot)
    return gt_onehot

class uncertainty_metrics():
    def __init__(self, L_bins, n_classes):
        self.L_bins = L_bins
        self.n_classes = n_classes
        self.conf_interval_size = 1.0 / L_bins
    
    def Brier_Score(self, probs, target):
        #target = np.where(probs[0, :, :] == 0.0, 0.0, target) #We do not consider the pixels with no prediction
        BS = np.sum((probs - target)**2, axis = 0)
        return np.mean(BS)
        
    def AUSE_metric(self, probs, target):
        probs = probs.astype(float) + 0.0000000001 #Avoid divide by zero
        BS = (np.sum((probs - target)**2, axis = 0)).flatten()
        entropy = (- np.sum(probs * np.log(probs), axis = 0)).flatten()

        total_predictions = BS.shape[0]
        sorted_inds_entropy = np.argsort(entropy)
        sorted_inds_error = np.argsort(BS) 
        entropy_brier_scores, error_brier_scores = [], []
        fractions = list(np.arange(start = 0.0, stop = 1.0, step = 0.05))

        for step, fraction in enumerate(fractions):
            entropy_brier_score = np.mean(BS[sorted_inds_entropy[0:int((1.0 - fraction) * total_predictions)]] )
            entropy_brier_scores.append(entropy_brier_score)
            error_brier_score = np.mean(BS[sorted_inds_error[0:int((1.0-fraction) * total_predictions)]])
            error_brier_scores.append(error_brier_score)
        
        error_brier_scores_normalized = error_brier_scores/error_brier_scores[0]
        entropy_brier_scores_normalized = entropy_brier_scores/error_brier_scores[0]
        sparsification_errors = entropy_brier_scores_normalized - error_brier_scores_normalized        
        ause = np.trapz(y=sparsification_errors, x=fractions)
        return {'ause': ause, 'bs': np.mean(BS)}
    
    def ECE_metric_v2(self, probs, label):
        interval_2_num_preds = {}
        interval_2_num_correct_preds = {}
        interval_2_confs = {}
        interval_2_mean_conf = {}
        for i in range(self.L_bins):
            interval_2_num_preds[i] = 0
            interval_2_num_correct_preds[i] = 0
            interval_2_confs[i] = np.array([])

        pred = (np.argmax(probs, axis=0).astype(np.uint8)).flatten() # (shape: (num_nonignores, ))
        conf = (np.max(probs, axis=0)).flatten() # (shape: (num_nonignores, ))
        label = label.flatten()
        #num_nonignores_predictions = np.count_nonzero(conf < 1.0)
        
        
        ACE, MCE = 0.0, 0.0
        reliability = []
        all_acc = []
        all_conf = []
        M = 0
        for i in range(self.L_bins):
            lower = i * self.conf_interval_size
            upper = (i + 1) * self.conf_interval_size
            y_pred = pred[np.nonzero(np.logical_and(conf >= lower, conf < upper))]
            y_label = label[np.nonzero(np.logical_and(conf >= lower, conf < upper))]
            #precision = metrics.precision_recall_fscore_support(y_label, y_pred, average='macro', zero_division = 0)[0]
            
            confs_in_interval = conf[np.nonzero(np.logical_and(conf >= lower, conf < upper))] # (shape: (num_preds_in_interval, ))
            num_preds_in_interval = confs_in_interval.shape[0]
            num_correct_preds_in_interval = np.count_nonzero(np.logical_and(np.logical_and(conf >= lower, conf < upper), pred == label))
            if num_preds_in_interval > 0:
                accuracy = float(float(num_correct_preds_in_interval)/float(num_preds_in_interval))
                confidence = np.mean(confs_in_interval)
                ACE = float(np.abs(confidence - accuracy))
                M += 1
                #ACE += float(float(num_preds_in_interval)/float(num_nonignores_predictions))*np.abs(accuracy - confidence)
                if np.abs(accuracy - confidence) > MCE:
                    MCE = np.abs(accuracy - confidence)
                reliability.append(accuracy)
            else:
                reliability.append(0.0)
        if not(M == 0):
            ACE = ACE / M
        
        reliability = [round(num, 3) for num in reliability]
        return {'ECE': ACE, 'MCE': MCE, 'reliability': reliability}
