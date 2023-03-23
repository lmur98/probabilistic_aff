import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_sample(sample):
    rgb_image = torch.permute(sample['img'], (1, 2, 0))
    landmarks = sample['obj_bbox']
    plt.imshow(rgb_image)
    for i in range(landmarks.shape[0]):
        down_left_corner = [landmarks[i, 0], landmarks[i, 1]] #To plot the bounding box
        w = -(landmarks[i, 0] - landmarks[i, 2])
        h = landmarks[i, 3] - landmarks[i, 1]
        #print('The coordinates of the object are', landmarks[i, 0, 0], landmarks[i, 0, 1])
        #plt.scatter(landmarks[i, :, 0], landmarks[i, :, 1], s=20, marker='+', c='r')
        plt.scatter(down_left_corner[0], down_left_corner[1],  s=20, marker='.', c='b')
        rect = Rectangle((down_left_corner[0], down_left_corner[1]), w, h, linewidth=1,edgecolor='r',facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)

def show_aff_label(sample):
    aff_number = sample['aff_map']
    h_img, w_img = aff_number.shape
    aff_map = np.zeros((h_img, w_img, 3))
    for i in range(h_img):
        for j in range(w_img):
            if aff_number[i,j] == 0: #Background, light blue
                aff_map[i,j,:] = [65, 237, 255]
            if aff_number[i,j] == 1: #Contain, red
                aff_map[i,j,:] = [255, 65, 65]
            if aff_number[i,j] == 2: #Cut, orange
                aff_map[i,j,:] = [255, 155, 0]
            if aff_number[i,j] == 3: #Display, dark blue
                aff_map[i,j,:] = [0, 40, 238]
            if aff_number[i,j] == 4: #Engine, brown
                aff_map[i,j,:] = [155, 116, 0]
            if aff_number[i,j] == 5: #Grasp, purple
                aff_map[i,j,:] = [238, 0, 246]  
            if aff_number[i,j] == 6: #Hit, grey
                aff_map[i,j,:] = [100, 100, 100] 
            if aff_number[i,j] == 7: #Pound, dark green
                aff_map[i,j,:] = [0, 160, 11]    
            if aff_number[i,j] == 8: #Support, blue-green
                aff_map[i,j,:] = [0, 155, 155] 
            if aff_number[i,j] == 9: #W-Grasp, pink 
                aff_map[i,j,:] = [255, 100, 255]

    landmarks = sample['obj_bbox']
    plt.imshow(aff_map.astype('uint8'))

    for i in range(landmarks.shape[0]):
        down_left_corner = [landmarks[i, 0], landmarks[i, 1]] #To plot the bounding box
        w = -(landmarks[i, 0] - landmarks[i, 2])
        h = landmarks[i, 3] - landmarks[i, 1]
        #plt.scatter(landmarks[i, :, 0], landmarks[i, :, 1], s=10, marker='.', c='r')
        plt.scatter(down_left_corner[0], down_left_corner[1],  s=10, marker='.', c='b')
        rect = Rectangle((down_left_corner[0], down_left_corner[1]), w, h, linewidth=1,edgecolor='r',facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
    