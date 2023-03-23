import os
import torch
import torch.utils.data as Data
import numpy as np
import cv2
from PIL import Image

from data.data_augmentation import my_compose, Rescale, RandomCrop, ToTensor, RandomFlip, Normalize, Pad, Input_noise

class IIAff_dataset(Data.Dataset):
    def __init__(self, dataset_path, mode): #Pad(32),Normalize([0.485, 0.456, 0.406]), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]                                                   
        self.img_dir = dataset_path + '/rgb/rgb'
        self.depth_dir = dataset_path + '/depth/depth'
        self.obj_dir = dataset_path + '/object_labels/object_labels'
        self.aff_dir = dataset_path + '/affordances_labels/affordances_labels'
        self.train_val_file = os.path.join(dataset_path, 'train_and_val.txt')
        self.test_prueba_file = os.path.join(dataset_path, 'test_prueba.txt')
        self.test_file = os.path.join(dataset_path, 'test.txt')
        self.show = os.path.join(dataset_path, 'show.txt')
        self.mode = mode
        self.transform = my_compose([ToTensor()])
        self.objects_dict = {0: 'bowl', 1: 'tvm', 2: 'pan', 3: 'hammer', 4: 'knife', 5: 'cup', 6: 'drill', 7: 'racket', 8: 'spatula', 9: 'bottle'}
        self.affordances_dict = {0: 'background', 1: 'contain', 2: 'cut', 3: 'display', 4: 'engine', 5: 'grasp', 6: 'hit', 7: 'pound', 8: 'support', 9: 'wrap grasp'}
        

    def read_indexer(self, indexer_file, idx):
        with open(indexer_file) as file:
            contents = file.readlines()
        return contents[idx].split('.')[0], len(contents)

    def __getitem__(self, index):
        if self.mode == 'training':
            sample, n_lines = self.read_indexer(self.train_val_file, index)
        elif self.mode == 'testing':
            sample, n_lines = self.read_indexer(self.test_file, index)
        elif self.mode == 'dropout':
            sample, n_lines = self.read_indexer(self.test_prueba_file, index)
        elif self.mode == 'show':
            sample, n_lines = self.read_indexer(self.show, index)
        else:
            print('Wrong mode, check the sintaxis')

        img_path = os.path.join(self.img_dir, sample + '.jpg')
        obj_path = os.path.join(self.obj_dir, sample + '.txt')
        aff_path = os.path.join(self.aff_dir, sample + '.txt')
        #depth_path = os.path.join(self.depth_dir, sample + '.txt')

        img = self.read_img(img_path)
        img_h, img_w, ch = img.shape
        #print('the initial shape is', img_h, img_w)
        aff_map = self.read_aff_map(aff_path)
        if img_h * img_w > 500*500:
            img = cv2.resize(img, (int(img_h / 2), int(img_w / 2)))
            aff_map = np.array(Image.fromarray(aff_map).resize((int(img_h / 2), int(img_w / 2)), Image.NEAREST)) #USE NEAREST INTERPOLATION, WE WORK WITH LABELS!! 
        
        obj_label, obj_bbox = self.get_bounding_box(obj_path)
        aff_label, aff_bboxes, aff_masks = self.get_aff_masks3(aff_map, obj_bbox)

        #aff_map_show = (self.read_aff_map(aff_path)).squeeze().flatten()
        #count_array = np.bincount(aff_map_show)
        #print('Original distribution of the affordances', count_array)

        #sample = {'img': img, 'depth': depth, 'aff_map': aff_map, 'obj_label': obj_label, 'obj_bbox': obj_bbox, 'obj_masks': obj_masks}
        target = {}
        target['boxes'] = aff_bboxes
        target['labels'] = aff_label
        target['masks'] = aff_masks
        target['aff_map'] = aff_map
        
        if self.transform:
            img, target = self.transform(img, target)

        return img, target, sample
    
    def read_img(self, img_filename):
        return cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)

    def read_aff_map(self, aff_txt):
        return  np.loadtxt(aff_txt).astype(np.uint8) #np.expand_dims(np.loadtxt(aff_txt).astype(int), axis = 2)
    
    def overlap_23(self, bb1, bb2):
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            cond = False
        else:
            cond = True
        return cond, [x_left, y_top, x_right, y_bottom]    

    def get_aff_masks(self, aff_map, obj_bbox):
        aff_label = []
        aff_masks = []
        aff_bboxes = []
        for i in range(obj_bbox.shape[0]):
            x1, y1, x2, y2 = obj_bbox[i, :]
            obj_aff = aff_map[y1:y2, x1:x2]
            ids = np.unique(obj_aff)
            ids = np.delete(ids, np.where(ids == 0)) # remove id 0 if it exists -> ignore BG
            for id in ids:
                G_AFF = np.zeros((aff_map.shape[0], aff_map.shape[1]), dtype = np.uint8)
                AFF = obj_aff.copy()
                AFF[AFF != id] = 0
                AFF[AFF == id] = 1
                G_AFF[y1:y2, x1:x2] = AFF

                # We avoid the overlapping of other object affordances!!!
                #for j in range(obj_bbox.shape[0]):
                #    if i != j: #We want to compare different objects!
                #        if self.overlap(obj_bbox[i], obj_bbox[j])[0]:
                #            x1_over, y1_over, x2_over, y2_over = self.overlap(obj_bbox[i], obj_bbox[j])[1]
                #            G_AFF[y1_over:y2_over, x1_over:x2_over] = np.zeros((y2_over - y1_over, x2_over - x1_over))
                #            #print('overlapping',x1_over, y1_over, x2_over, y2_over)
        
                a = np.where(G_AFF != 0)
                if a[0].shape[0] == 0:
                    proposed_coord = x1, y1, x2, y2
                    proposed_label = 0
                else:
                    #x1_aff, y1_aff, x2_aff, y2_aff = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
                    #proposed_coord = x1_aff, y1_aff, x2_aff, y2_aff
                    proposed_coord = x1, y1, x2, y2
                    proposed_label = id
                    
                #We want to avoid zero bounding boxes
                if (proposed_coord[2] > proposed_coord[0] + 3) and (proposed_coord[3] > proposed_coord[1] + 3):
                    aff_bboxes.append([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2]), int(proposed_coord[3])])
                    aff_label.append(int(proposed_label))
                    aff_masks.append(G_AFF)
                else:
                    aff_bboxes.append([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2] + 3), int(proposed_coord[3] + 3)])
                    aff_label.append(int(id))
                    aff_masks.append(G_AFF)

        if len(aff_label) == 0:
            aff_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            aff_label.append(int(0))
            aff_masks.append(np.zeros((aff_map.shape[0], aff_map.shape[1]), dtype = np.uint8))

        #print('THE SHAPE OF THE LABELS IS', torch.Tensor(aff_label).shape, torch.Tensor(aff_label).dtype)
        #print('SHAPE OF THE BBOXES', np.array(aff_bboxes).shape, np.array(aff_bboxes).dtype)
        #print('SHAPE OF MASKS', len(aff_masks), aff_masks[0].shape, aff_masks[0].dtype)

        return torch.Tensor(aff_label), np.array(aff_bboxes), aff_masks
    
    def get_object_masks(self, aff_map, obj_label, obj_bbox):
        mask = np.vectorize(lambda x: x in {1, 2, 3, 4, 5, 6, 7, 8, 9})
        all_objects_masks = []
        for i in range(len(obj_label)): #We create masks for the objects, not for the affordances. Maybe there are two affordances in an object. I.E: cup -> contain, grasp
            object_mask = np.zeros_like(aff_map)
            x_min, y_min, x_max, y_max = obj_bbox[i]
            #x_max, y_max = obj_bbox[i, 1]
            silouted_crop = aff_map[y_min:y_max, x_min:x_max]
            silouted_crop = mask(silouted_crop).astype(int)
            #print('The silouted crop has a shape of', silouted_crop.shape, 
            #      'it is a', self.objects_dict[obj_label[i].item()],
            #      'and it is in this coordinates x_min', x_min, 'x_max', x_max, 'y_min', y_min, 'ymax', y_max)
            object_mask[y_min:y_max, x_min:x_max] = silouted_crop
            all_objects_masks.append(object_mask)
        return all_objects_masks

    def get_bounding_box(self, bbox_file_path):
        all_objects = []
        all_bbox = []
        with open(bbox_file_path) as f:
            for line in f:
                obj_id, x_min, y_min, x_max, y_max = line.split()
                all_objects.append(int(obj_id))
                all_bbox.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                #print('The object is', self.objects_dict[int(obj_id)], 'the true xmin is', x_min, 'x_max is', x_max, 'ymin is', y_min, 'ymax', y_max)
        return torch.Tensor(all_objects), np.array(all_bbox) 

    def separate_by_object(self, aff_map, aff_id, obj_coords):
        x1, y1, x2, y2 = obj_coords
        obj_aff = aff_map[y1:y2, x1:x2]

        ids_i = np.unique(obj_aff)
        ids_i = np.delete(ids_i, np.where(ids_i == 0)) # remove id 0 if it exists -> ignore BG
        for obj_aff_id in ids_i:
            if obj_aff_id == aff_id: #Con esto lo contamos solo una vez
                G_AFF = np.zeros((aff_map.shape[0], aff_map.shape[1]), dtype = np.uint8)

                AFF = obj_aff.copy()
                AFF[AFF != aff_id] = 0
                AFF[AFF == aff_id] = 1
                G_AFF[y1:y2, x1:x2] = AFF
                return G_AFF

    def extract_metrics_G_AFF(self, G_AFF_obj, aff_id):
        a = np.where(G_AFF_obj != 0)
        new_a_x = a[1] #X-coordinates where there is that instance
        new_a_y = a[0] #Y-coordinates where there is that instance

        x1_aff, x2_aff = np.min(new_a_x), np.max(new_a_x)
        y1_aff, y2_aff = np.min(new_a_y), np.max(new_a_y)
        

        proposed_coord = x1_aff, y1_aff, x2_aff, y2_aff
        proposed_label = aff_id
        final_aff_map = G_AFF_obj

        return proposed_coord, proposed_label, final_aff_map

    def overlap(self, bb1, bb2, G_AFF, aff_id):
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            cond = False
        else:
            cond = True
        return cond

    def make_zeros_overlap(self, bb1, bb2, G_AFF, aff_id):
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if self.count_aff_pixels(bb1, bb2, G_AFF, aff_id):
            G_AFF[y_top:y_bottom, x_left:x_right] = np.zeros((y_bottom - y_top, x_right - x_left))
        return G_AFF

    def count_aff_pixels(self, bb1, bb2, G_AFF, aff_id):
        G_AFF_1 = G_AFF[bb1[1]:bb1[3], bb1[0]:bb1[2]]
        G_AFF_2 = G_AFF[bb2[1]:bb2[3], bb2[0]:bb2[2]]
        aff_1 = np.count_nonzero(G_AFF_1 == aff_id)
        aff_2 = np.count_nonzero(G_AFF_2 == aff_id)
        if aff_1 > aff_2:
            remove = False
        else:
            remove = True
        return remove

    def get_aff_masks2(self, aff_map, obj_bbox):
        'We change the scheme: now we iterate through the affordances and then we sepparate if they belong to different objects'
        aff_label = []
        aff_masks = []
        aff_bboxes = []
        all_aff_ids = np.unique(aff_map)
        all_aff_ids = np.delete(all_aff_ids, np.where(all_aff_ids == 0))
        if len(all_aff_ids) != 0:
            for aff_id in all_aff_ids:
                G_AFF = aff_map.copy()
                for obj_i in range(obj_bbox.shape[0]):
                    #If there is overlap we modify the aff map
                    for obj_j in range(obj_bbox.shape[0]):
                        if obj_i != obj_j:
                            if self.overlap(obj_bbox[obj_i], obj_bbox[obj_j], G_AFF, aff_id):
                                G_AFF = self.make_zeros_overlap(obj_bbox[obj_i], obj_bbox[obj_j], aff_map.copy(), aff_id)

                    G_AFF_obj = self.separate_by_object(G_AFF, aff_id, obj_bbox[obj_i, :])
                    if G_AFF_obj is not None:
                        proposed_coord, proposed_label, final_aff_map = self.extract_metrics_G_AFF(G_AFF_obj, aff_id)
                        if (proposed_coord[2] > proposed_coord[0] + 5) and (proposed_coord[3] > proposed_coord[1] + 5):
                            aff_bboxes.append([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2]), int(proposed_coord[3])])
                            aff_label.append(int(proposed_label))
                            aff_masks.append(final_aff_map)
                        else:
                            aff_bboxes.append([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2] + 5), int(proposed_coord[3] + 5)])
                            aff_label.append(int(proposed_label))
                            aff_masks.append(final_aff_map)
 
        if len(aff_bboxes) == 0:
            proposed_coord = obj_bbox[0, :]
            #print([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2]), int(proposed_coord[3])])
            proposed_label = 0
            final_aff_map = np.zeros((aff_map.shape[0], aff_map.shape[1]), dtype = np.uint8)
            aff_bboxes.append([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2]), int(proposed_coord[3])])
            aff_label.append(proposed_label)
            aff_masks.append(final_aff_map)
        
        
        return torch.Tensor(aff_label), np.array(aff_bboxes), aff_masks

    def get_aff_masks3(self, aff_map, obj_bbox):
        aff_label = []
        aff_masks = []
        aff_bboxes = []
        all_aff_ids = np.unique(aff_map)
        all_aff_ids = np.delete(all_aff_ids, np.where(all_aff_ids == 0))
        if len(all_aff_ids) != 0:
            for aff_id in all_aff_ids:
                G_AFF = aff_map.copy()
                zeros = np.zeros_like(G_AFF)
                ones = np.ones_like(G_AFF)
                gray = np.where(G_AFF == aff_id, ones, zeros) * 255 #We create a binary map for each affordance
                
                # We compute the contours of the different affordances in order to separate them
                thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                for cntr in contours:
                    x,y,w,h = cv2.boundingRect(cntr)
                    if w < 2:
                        w = w + 3
                    if h < 2:
                        h = h + 3
                    aff_bboxes.append([int(x), int(y), int(x+w), int(y+h)])
                    aff_label.append(aff_id)
                    mask_aff_obj = np.zeros_like(G_AFF)
                    mask_aff_obj[y:y+h, x:x+w] = gray[y:y+h, x:x+w] / 255
                    aff_masks.append(mask_aff_obj)

        else:
            proposed_coord = obj_bbox[0, :]
            proposed_label = 0
            final_aff_map = np.zeros((aff_map.shape[0], aff_map.shape[1]), dtype = np.uint8)

            aff_bboxes.append([int(proposed_coord[0]), int(proposed_coord[1]), int(proposed_coord[2]), int(proposed_coord[3])])
            aff_label.append(proposed_label)
            aff_masks.append(final_aff_map)
        
             
        return torch.Tensor(aff_label), np.array(aff_bboxes), aff_masks

    
    def __len__(self):
        if self.mode == 'training':
            with open(self.train_val_file) as file:
                return len(file.readlines())
        elif self.mode == 'testing':
            with open(self.test_file) as file:
                return len(file.readlines())
        elif self.mode == 'dropout':
            with open(self.test_prueba_file) as file:
                return len(file.readlines())
        elif self.mode == 'show':
            with open(self.show) as file:
                return len(file.readlines())

def my_collate(batch):
    imgs_batch = list()
    targets_batch = list()
    samples_name_batch = list()
    
    for b in batch:
        imgs_batch.append(b[0])
        targets_batch.append(b[1])
        samples_name_batch.append(b[2])

    return imgs_batch, targets_batch, samples_name_batch