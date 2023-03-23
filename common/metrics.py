from typing import final
import numpy as np
import scipy
import scipy.ndimage
from PIL import Image
import torch
import random
import os
import cv2

from data.data_augmentation import transform_img
from scipy.ndimage import distance_transform_edt, gaussian_filter

def get_iou_bboxes(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def filter_detections_v2(gt_mask, gt_bboxes, gt_label, pred_mask, pred_bboxes, pred_labels_prob, pred_sigmoid):
    # https://github.com/beapc18/AffordanceNet/blob/main/utils/eval_utils.py
    pred_label = np.argmax(pred_labels_prob, axis = 1)
    if pred_bboxes.shape[0] > 0:
        filter_pred_masks = []
        filter_labels = []
        filter_sigm_mask = []
        filter_bboxes = []
        filter_labels_probs = []
        for obs in range(pred_bboxes.shape[0]):
            max_IoU = 0 #We have to filter the different observations!!
            for box in range(gt_bboxes.shape[0]):
                IoU = get_iou_bboxes(gt_bboxes[box, :], pred_bboxes[obs, :])
                if IoU > max_IoU:
                    max_IoU = IoU
                    max_IoU_gt_label = gt_label[box]
                    max_IoU_pred_label = pred_label[obs]
            if max_IoU > 0.3 and (max_IoU_gt_label == max_IoU_pred_label): #Threshold in the paper!
                filter_pred_masks.append(pred_mask[obs, :, :, :])
                filter_labels.append(max_IoU_pred_label) 
                filter_sigm_mask.append(pred_sigmoid[obs, :, :, :])
                filter_bboxes.append(pred_bboxes[obs, :])
                filter_labels_probs.append(pred_labels_prob[obs, :])
                
        if len(filter_labels) != 0:
            filter_pred_masks = np.asarray(filter_pred_masks)
            filter_labels = np.asarray(filter_labels)
            filter_sigm_mask = np.asarray(filter_sigm_mask)
            filter_labels_probs = np.asarray(filter_labels_probs)
            filter_bboxes = np.asarray(filter_bboxes)
        else:
            filter_pred_masks = None
            filter_labels = None
            filter_sigm_mask = None
            filter_labels_probs = None
            filter_bboxes = None
    else: 
        filter_pred_masks = None
        filter_labels = None
        filter_sigm_mask = None
        filter_labels_probs = None
        filter_bboxes = None

    return filter_pred_masks, filter_sigm_mask, filter_bboxes, filter_labels_probs

def filter_detections(gt_mask, gt_bboxes, gt_label, pred_mask, pred_bboxes, pred_label):
    # https://github.com/beapc18/AffordanceNet/blob/main/utils/eval_utils.py
    
    if pred_bboxes.shape[0] > 0:
        filter_pred_masks = []
        filter_labels = []
        for obs in range(pred_bboxes.shape[0]):
            max_IoU = 0 #We have to filter the different observations!!
            for box in range(gt_bboxes.shape[0]):
                IoU = get_iou_bboxes(gt_bboxes[box, :], pred_bboxes[obs, :])
                if IoU > max_IoU:
                    max_IoU = IoU
                    max_IoU_gt_label = gt_label[box]
                    max_IoU_pred_label = pred_label[obs]
            if max_IoU > 0.3 and (max_IoU_gt_label == max_IoU_pred_label): #Threshold in the paper!
                filter_pred_masks.append(pred_mask[obs, :, :, :])
                filter_labels.append(max_IoU_pred_label) 
        if len(filter_labels) != 0:
            filter_pred_masks = np.asarray(filter_pred_masks)
            filter_labels = np.asarray(filter_labels)
        else:
            filter_pred_masks = None
            filter_labels = None
    else: 
        filter_pred_masks = None
        filter_labels = None

    return filter_pred_masks, filter_labels

def group_detections(pred_masks, pred_labels):
    if pred_masks is not None:
        pred = np.squeeze(pred_masks, axis = 1) #(masks, h, w)
        sorted_pred = np.zeros_like(pred_masks)
        for obs in range(pred.shape[0]):
            sorted_pred[obs, :, :] = np.where(pred[obs, :, :] > 0.5, pred_labels[obs], 0)
        global_pred = np.squeeze(np.max(sorted_pred, axis = 0).astype(int))
    else:
        global_pred = None

    return global_pred

def update_stats_affordances(stats, final_gt_mask, global_pred):
    """
        Updates stats for affordance detection.
        Calculates FwB measurement for the whole image that contains all the masks for all the objects.
        https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/resources/WFb.m
    :param stats: accumulated statistics for affordance evaluation
    :param final_gt_mask: mask with all gt masks in the image
    :param final_pred_mask: mask with all predicted masks in the image #final_pred_mask = np.argmax(final_pred, axis = 0)
    :returns: json with updated stats for affordance detection
    """
    if global_pred is None:
        global_pred = np.zeros((final_gt_mask.shape[0], final_gt_mask.shape[1])) #We set everything to background

    ids = np.unique(final_gt_mask)
    ids = np.delete(ids, np.where(ids == 0)) # remove id 0 if it exists -> ignore BG
    eps = np.finfo(float).eps

    
    for id in ids:
        # separate BG and FG in gt mask and predicted mask (0 for bg, 1 for class)
        G = final_gt_mask.copy()
        G[G != id] = 0
        G[G == id] = 1

        D = global_pred.copy()
        D[D != id] = 0
        D[D == id] = 1
        D = D.astype(np.float)
        E = np.abs(G - D)

        # Calculate Euclidean distance for each pixel to the closest FG pixel and closest pixel
        # (logical not because the function calculates distance to BG pixels)
        dist, dist_idx = distance_transform_edt(np.logical_not(G), return_indices=True)

        # Pixel dependency
        Et = E.copy()

        # Replace in Et where G is 0 by the value in Et[idxt] where G is 0
        x = dist_idx[0][G == 0]
        y = dist_idx[1][G == 0]

        Et[G == 0] = Et[x, y]

        # calculate truncate is necessary if we want to fix the kernel size
        sigma, window_size = 5, 7
        t = (((window_size - 1) / 2) - 0.5) / sigma
        EA = gaussian_filter(Et, sigma, truncate=t)
        min_E_EA = E.copy()
        min_E_EA[np.logical_and(G == 1, EA < E)] = EA[np.logical_and(G == 1, EA < E)]

        # Pixel importance
        B = np.ones(final_gt_mask.shape)
        B[G != 1] = 2 - np.exp((np.log(0.5) / 5.0) * dist[G != 1])

        Ew = min_E_EA * B

        tp = np.sum((1 - Ew) * G)
        fp = np.sum(Ew * (1 - G))
        r = 1 - np.mean(Ew[G == 1])
        p = tp / (tp + fp + eps)
        q = 2 * r * p / (r + p + eps)
    
        stats[id]["q"].append(q)
    return stats

def init_stats(labels):
    """
        Initializes the statistics.
    :param labels: array with all possible labels
    :returns: json with empty statistics for all labels
    "tp": [], "total": 0,
            "fp": [],
            "scores": [],
    """
    stats = {}
    for i, label in enumerate(labels):
        if i == 0:
            continue
        stats[i] = {
            "label": label,
            "q": []
        }
    return stats

def init_loss_stats():
    stats = {'loss_mask': [],'loss_classifier': [], 'loss_box_reg': [], 'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_total': []}
    return stats

def remap_to_show(pred, gt, index):
    pred_img = np.zeros((pred.shape[0], pred.shape[1], 3))
    gt_img = np.zeros((pred.shape[0], pred.shape[1], 3))
    dict_aff = {0: [65, 237, 255], 1:[255, 65, 65], 2:[255, 155, 0], 3:[0, 40, 238], 4:[155, 116, 0], 5:[255, 160, 255], 6:[100, 100, 100] , 7:[0, 160, 11], 8:[127, 0, 255], 9:[255, 255, 0]}
    for old, new in dict_aff.items():
        pred_img[pred == old] = new
        gt_img[gt == old] = new
 
    pred_im_show = Image.fromarray(pred_img.astype('uint8'))
    pred_im_show.save('/home/lmur/documents/iit_aff_DATASET/qualitative_results/'+str(index)+'_pred_2.jpeg')

    gt_im_show = Image.fromarray(gt_img.astype('uint8'))
    gt_im_show.save('/home/lmur/documents/iit_aff_DATASET/qualitative_results/'+str(index)+'_gt_2.jpeg')

    img = show_brg_img(index)
    cv2.imwrite('/home/lmur/documents/iit_aff_DATASET/qualitative_results/'+str(index)+'_img.jpeg', img)

def show_brg_img(index):
    img_path = os.path.join('/home/lmur/documents/iit_aff_DATASET/rgb/rgb', index + '.jpg')
    img_show_brg = (cv2.imread(img_path)).astype(np.uint8).copy()
    img_h, img_w, ch = img_show_brg.shape
    if img_h * img_w > 500*500:
        img_show_brg = cv2.resize(img_show_brg, (int(img_h / 2), int(img_w / 2)))
    return img_show_brg

def show_all_bboxes(bboxes, index, detections):
    img_show_brg = show_brg_img(index)
    for b in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[b, :]
        cv2.rectangle(img_show_brg,(int(x1), int(y1)),(int(x2), int(y2)), (0,255,0), 1) #Green
    cv2.imwrite('/home/lmur/Documents/iit_aff/qualitative_results/'+str(index)+'_img_observed_bboxes.jpeg', img_show_brg)
    img_show_brg = show_brg_img(index)
    for d in range(len(detections)):
        x1, y1, x2, y2 = np.mean(detections[d], axis = 1)
        cv2.rectangle(img_show_brg,(int(x1.item()), int(y1.item())),(int(x2.item()), int(y2.item())), (0,0,255),2) #Red
    cv2.imwrite('/home/lmur/Documents/iit_aff/qualitative_results/'+str(index)+'_img_detected_bboxes.jpeg', img_show_brg)

def show_all_masks(masks, index, detections):
    img_show_brg = show_brg_img(index)
    for b in range(masks.shape[0]):
        color = np.array([0,255,0], dtype = 'uint8')
        masked_img = np.squeeze(np.where((masks[b] > 0.5)[...,None], color, img_show_brg)) #The detections in green!
        img_show_brg = cv2.addWeighted(img_show_brg, 0.6, masked_img, 0.4,0)
    for d in range(len(detections)):
        color = np.array([0, 0, 255], dtype = 'uint8')
        detection_sigm = np.mean(detections[d], axis = 0)
        masked_detection = np.squeeze(np.where((detection_sigm > 0.5)[...,None], color, img_show_brg))
        img_show_brg = cv2.addWeighted(img_show_brg, 0.2, masked_detection, 0.8,0)
    cv2.imwrite('/home/lmur/Documents/iit_aff/qualitative_results/'+str(index)+'_img_masks.jpeg', img_show_brg)

def show_variance(index, detections, bboxes, label_var):
    trace_sum = np.zeros_like(detections[0]['trace'])
    """
    for d in range(len(detections)):
        trace_sum = ldetections[d]['trace'] #Choose this parameter to add the label var detections[d]['trace'] +
        trace_max = np.max(trace_sum)
        img_show_brg = show_brg_img(index)
        trace = (trace_sum * 255 / trace_max).astype(np.uint8)
        imC = cv2.applyColorMap(trace, cv2.COLORMAP_JET)
        img_show_brg = cv2.addWeighted(img_show_brg, 0.5, imC, 0.5,0)
        cv2.imwrite('/home/lmur/documents/iit_aff_DATASET/qualitative_results/'+str(index)+'_'+str(d)+'_img_var_masks.jpeg', img_show_brg)
    """
    for d in range(len(detections)):
        bbox = bboxes[d]
        trace_sum += detections[d]['trace'] #+ label_var[d]
    trace_max = np.max(trace_sum)
    img_show_brg = show_brg_img(index)
    trace = (trace_sum * 255 / trace_max).astype(np.uint8)
    imC = cv2.applyColorMap(trace, cv2.COLORMAP_JET)
    img_show_brg = cv2.addWeighted(img_show_brg, 0.5, imC, 0.5,0)
    cv2.imwrite('/home/lmur/documents/iit_aff_DATASET/qualitative_results/'+str(index)+'spatial_var_epistemic.jpeg', img_show_brg)
    


def compute_epoch_loss(loss_stats):
    epoch_stats = {'loss_classifier': [], 'loss_box_reg': [], 'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_total': []}
    for key, value in loss_stats.items():
        epoch_stats[key] = np.mean(value)
    return epoch_stats

def compute_mean_Fb(no_filt_Fb_weighted_score, Fb_weighted_score, labels_dict):
    filter_mean_Fb = 0
    no_filt_mean_Fb = 0
    for aff_class in range(len(Fb_weighted_score)):
        if len(Fb_weighted_score[aff_class + 1]['q']) > 0:
            filter_mean_Fb += np.mean(Fb_weighted_score[aff_class + 1]['q'])
            no_filt_mean_Fb += np.mean(no_filt_Fb_weighted_score[aff_class + 1]['q'])
            #print('estamos acumulando en',len(Fb_weighted_score[aff_class + 1]['q']), 'para la clase', aff_class)
        else:
            filter_mean_Fb += 0
            no_filt_mean_Fb += 0
        print('The filter F_b metric is', np.mean(Fb_weighted_score[aff_class + 1]['q']), 'for the class', labels_dict[aff_class + 1])
        print('The no filtered F_b metric is', np.mean(no_filt_Fb_weighted_score[aff_class + 1]['q']), 'for the class', labels_dict[aff_class + 1])
    filter_mean_Fb /= 9 #Number of affordances
    no_filt_mean_Fb /= 9
    print('AND THE FILTER MEAN FB IS', filter_mean_Fb, 'AND THE NO FILT MEAN', no_filt_mean_Fb)
    return filter_mean_Fb, no_filt_mean_Fb

def compute_mean_Fb_v2(Fb_weighted_score, labels_dict):
    filter_mean_Fb = 0
    Fb_class =  []
    for aff_class in range(len(Fb_weighted_score)):
        if len(Fb_weighted_score[aff_class + 1]['q']) > 0:
            filter_mean_Fb += np.mean(Fb_weighted_score[aff_class + 1]['q'])
        else:
            filter_mean_Fb += 0
        Fb_class.append(100*np.mean(Fb_weighted_score[aff_class + 1]['q']))
        #print('The F_b metric is', format(100 * np.mean(Fb_weighted_score[aff_class + 1]['q']), ".2f"), 'for the class', labels_dict[aff_class + 1])
    Fb_class = [round(num, 3) for num in Fb_class]
    print(Fb_class)
    filter_mean_Fb /= 9 #Number of affordances
    print('AND THE FILTER MEAN FB IS', format(100 * filter_mean_Fb, ".3f"))
    return filter_mean_Fb