import numpy as np
import time
from scipy.special import expit


def get_iou_bboxes(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    #assert bb1['x1'] < bb1['x2']
    #assert bb1['y1'] < bb1['y2']
    #assert bb2['x1'] < bb2['x2']
    #assert bb2['y1'] < bb2['y2']

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

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_masks(mask1, mask2):
    mask1 = np.squeeze(np.where(mask1 > 0.5, 1, 0))
    mask2 = np.squeeze(np.where(mask2 > 0.5, 1, 0))
    mask1_area = np.count_nonzero(mask1 == 1)    
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    if mask1_area == 0 and mask2_area == 0:
        iou = 0.0
    else:
        iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

def compute_final_prediction(detected_prob, detected_bbox, detected_label):
    #We iterate through the number of detections
    mean_bboxes, all_mean_labels, all_prob_labels = [], [], []
    for d in range(len(detected_prob)):
        probs = detected_prob[d]
        reduced = np.mean(probs, axis = 0)
        if d == 0:
            all_probs_obs = np.expand_dims(reduced, axis = 0)
        else:
            all_probs_obs = np.concatenate((all_probs_obs, np.expand_dims(reduced, axis = 0)), axis = 0)
        mean_bboxes.append(np.mean(detected_bbox[d], axis = 1))
        mean_label = np.mean(detected_label[d], axis = 0) #We compute the mean of the class logits
        all_prob_labels.append(mean_label)
        all_mean_labels.append(np.argmax(mean_label))
        print(all_probs_obs.shape, 'shape of the probability maps clustered')
    return np.asarray(mean_bboxes), np.asarray(all_mean_labels), all_probs_obs, np.asarray(all_prob_labels)


def cluster_vectors(observations, obs_probs, obs_labels, obs_bboxes):
    theta_range = 0.5 #Extracted from the paper "Evaluating Merging Strategies for Sampling-based Uncertainty Techniques in Object Detection" https://arxiv.org/pdf/1809.06006.pdf
    clusters_sigm = []  # List of representative vectors
    clusters_sigm.append(observations[0])  # Initialization of cluster 1 with the first vector
    #clusters_probs = []
    #clusters_probs.append(np.expand_dims(obs_probs[0], axis = 0))
    #clusters_label = []
    #clusters_label.append(np.expand_dims(obs_labels[0], axis = 0))
    clusters_bbox_group = []
    clusters_bbox_group.append(np.expand_dims(obs_bboxes[0], axis = 1))
    clusters_index = []
    clusters_index.append(np.array([0]))
    pred_labels = np.argmax(obs_labels, axis = 1)
    clusters_label = []
    clusters_label.append(np.array([pred_labels[0]]))

    # Cluster the rest of the vectors
    for i in range(1, observations.shape[0]):
        forms_exiting = False
        for j in range(len(clusters_bbox_group)):
            mean_pos_cluster = np.sum(clusters_sigm[j], axis = 0) / len(clusters_sigm[j])
            IoU = get_iou_masks(observations[i], mean_pos_cluster)
            #mean_pos_cluster = np.mean(clusters_bbox_group[j], axis = 1)
            #IoU = get_iou_bboxes(obs_bboxes[i], mean_pos_cluster)

            if (IoU > theta_range) and (clusters_label[j][0] == pred_labels[i]): #and (clusters_label[j] == obs_labels[i]): #They match, so they join an existing cluster
                forms_exiting = True
                clusters_index[j] = np.concatenate((clusters_index[j], np.array([i])), axis = 0)
                clusters_sigm[j] = np.concatenate((clusters_sigm[j], observations[i]), axis = 0) #.append(observations[i])
                clusters_bbox_group[j] = np.concatenate((clusters_bbox_group[j], np.expand_dims(obs_bboxes[i], axis = 1)), axis = 1)
                clusters_label[j] = np.concatenate((clusters_label[j], np.array([pred_labels[i]])), axis = 0)
                #clusters_probs[j] = np.concatenate((clusters_probs[j], np.expand_dims(obs_probs[i], axis = 0)), axis = 0)
                #clusters_label[j] = np.concatenate((clusters_label[j], np.expand_dims(obs_labels[i], axis = 0)), axis = 0)
        if not(forms_exiting): #Form a new cluster with that label
            clusters_index.append(np.array([i]))
            clusters_sigm.append(observations[i])
            clusters_bbox_group.append(np.expand_dims(obs_bboxes[i], axis = 1))
            clusters_label.append(np.array([pred_labels[i]]))
            #clusters_probs.append(np.expand_dims(obs_probs[i], axis = 0))
            #clusters_label.append(np.expand_dims(pred_labels[i], axis = 0))
    
    clusters_bbox, clusters_probs, clusters_label = [], [], []
    for c in range(len(clusters_index)):
        indexs = clusters_index[c]
        clusters_bbox.append(np.transpose(obs_bboxes[indexs]))
        clusters_label.append(obs_labels[indexs])
        clusters_probs.append(obs_probs[indexs])
    
    return clusters_probs, clusters_bbox, clusters_label, clusters_sigm #List of K elements (number of detections) of (n, w, h); where n is the number of observations of that detections

def compute_variance(detections):
    filter_detections = []
    classes = 10
    
    for i in range(len(detections)):
        print('aaaanother detection', detections[i].shape)
        print(np.unique(np.sum(detections[i], axis = 1)))
        #detections[i] is a vector of (N_observations, classes, H, W)
        epist_sum = np.zeros((detections[i][0].shape[1], detections[i][0].shape[2], classes, classes)) #(H,W,cls,cls)
        aleat_sum = np.zeros((detections[i][0].shape[1], detections[i][0].shape[2], classes, classes)) #(H,W,cls,cls)
        diag = np.zeros((detections[i][0].shape[1], detections[i][0].shape[2], classes, classes)) #(H,W,cls,cls)
        
        p_t_bar = np.expand_dims(np.mean(detections[i], axis = 0), axis = 1)
        print(p_t_bar.shape, 'p_t_bar')
        p_t_bar = np.moveaxis(p_t_bar, (0, 1, 2, 3), (2, 3, 0, 1))
        for t in range(detections[i].shape[0]):    
            p_t_hat = np.expand_dims(detections[i][t], axis = 1)
            values, counts = np.unique(p_t_hat, return_counts=True)
            p_t_hat = np.moveaxis(p_t_hat, (0, 1, 2, 3), (2, 3, 0, 1)) #(H, W, classes, 1)
            epist = (p_t_hat - p_t_bar) @ np.transpose((p_t_hat - p_t_bar), (0, 1, 3, 2)) #(H, W, classes, clases)
            for c in range(p_t_hat.shape[2]):
                diag[:, :, c, c] = p_t_hat[:, :, c, 0]
            aleat = diag - np.matmul((p_t_hat), np.transpose(p_t_hat, (0, 1, 3, 2)))
            epist_sum = np.add(epist_sum, epist, out=epist_sum, casting="unsafe")
            aleat_sum = np.add(aleat_sum, aleat, out=aleat_sum, casting="unsafe")
        print(np.unique((aleat_sum / detections[i].shape[0])), 'unique de la aleat')
        variance_matrix = (epist_sum / detections[i].shape[0]) + (aleat_sum / detections[i].shape[0]) 
        trace = np.trace(variance_matrix, axis1 = 2, axis2 = 3)
        mean = np.mean(detections[i], axis = 0)
        dict_detection = {'mean': mean,  #(cls, H, W)
                          'epistemic': epist_sum / detections[i].shape[0], #(H, W, cls, cls)
                          'aleatoric': aleat_sum / detections[i].shape[0], #(H, W, cls, cls)
                          'trace': trace} #(H, W)
        filter_detections.append(dict_detection)

    return filter_detections

def compute_binary_variance(detections):
    filter_detections = []
    classes = 10
    for i in range(len(detections)):
        epist_sum = np.zeros((detections[i].shape[1], detections[i].shape[2])).astype('float64') #(H,W)
        aleat_sum = np.zeros((detections[i].shape[1], detections[i].shape[2])).astype('float64') #(H,W)
        p_t_mean = np.mean(detections[i], axis = 0).astype('float64') #(classes, 1)
        for t in range(detections[i].shape[0]):
            p_t = detections[i][t].astype('float64') 
            epist = (p_t - p_t_mean) * ((p_t - p_t_mean))
            aleat = p_t - p_t * p_t
            epist_sum += epist
            aleat_sum += aleat
        epist_var = epist_sum / detections[i].shape[0]
        aleat_var = aleat_sum / detections[i].shape[0]
        variance_matrix = epist_var
        dict_detection = {'mean': p_t_mean,  #(cls, H, W)
                          'epistemic': epist_var, #(H, W, cls, cls)
                          'aleatoric': aleat_var, #(H, W, cls, cls)
                          'trace': variance_matrix} #(H, W)
        filter_detections.append(dict_detection)
    return filter_detections 

    
def compute_variance_label(detections):
    filter_detections = []
    classes = 10
    
    for i in range(len(detections)): #detections[i] is a vector of (N_observations, classes)
        epist_sum = np.zeros((classes, classes)) #(cls,cls)
        aleat_sum = np.zeros((classes, classes)) #(cls,cls)
        diag = np.zeros((classes, classes)) #(H,W,cls,cls)
        p_t_bar = np.expand_dims(np.mean(detections[i], axis = 0), axis = 1)  #(classes, 1)
        for t in range(detections[i].shape[0]):    
            p_t_hat = np.expand_dims(detections[i][t], axis = 1) #(classes, 1)
            epist = (p_t_hat - p_t_bar) @ np.transpose((p_t_hat - p_t_bar)) #(classes, clases)
            for c in range(p_t_hat.shape[0]):
                diag[c, c] = p_t_hat[c, 0]
            aleat = diag - np.matmul((p_t_hat), np.transpose(p_t_hat))
            epist_sum = np.add(epist_sum, epist, out=epist_sum, casting="unsafe")
            aleat_sum = np.add(aleat_sum, aleat, out=aleat_sum, casting="unsafe")
        variance_matrix = (epist_sum / detections[i].shape[0]) + (aleat_sum / detections[i].shape[0]) #(epist_sum / detections[i].shape[0]) +
        total_trace = np.trace(variance_matrix, axis1 = 0, axis2 = 1)
        dict_detection = {'epistemic': epist_sum / detections[i].shape[0], 'aleatoric': aleat_sum / detections[i].shape[0], 'trace': total_trace} 
        filter_detections.append(dict_detection)
    return filter_detections