import torch
import numpy as np
#import skimage
#from skimage import io, transform
from PIL import Image
import random
#import mmcv


class my_compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class transform_img(object):
    def __init__(self):
        self.transforms = [Rescale_img((1330, 795)), Pad_img(32), ToTensor_img()]

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class Rescale_img(object):
    'Rescale the image in a sample to a given size'
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, image):
        #Resize image
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


class Rescale(object):
    'Rescale the image in a sample to a given size'
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, image, target):
        obj_bbox, obj_label, obj_masks, aff_map = target['boxes'], target['labels'], target['masks'], target['aff_map']
        
        #Resize image
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        #depth = transform.resize(depth, (new_h, new_w))
        aff_map = np.array(Image.fromarray(aff_map).resize((new_w, new_h), Image.NEAREST)) #USE NEAREST INTERPOLATION, WE WORK WITH LABELS!! 
        reshape_obj_masks = []
        for i in range(len(obj_masks)):
            obj_masks_i = np.array(Image.fromarray(obj_masks[i]).resize((new_w, new_h), Image.NEAREST))
            reshape_obj_masks.append(obj_masks_i)

        obj_bbox[:, 0] = obj_bbox[:, 0] * [new_w/w]
        obj_bbox[:, 1] = obj_bbox[:, 1] * [new_h/h]
        obj_bbox[:, 2] = obj_bbox[:, 2] * [new_w/w]
        obj_bbox[:, 3] = obj_bbox[:, 3] * [new_h/h]

        #aff_map_show = aff_map.squeeze().flatten().astype(int)
        #count_array = np.bincount(aff_map_show)
        #print('Distribution of affordances after the reshape',count_array)
        target_out = {}
        target_out['boxes'] = obj_bbox
        target_out['labels'] = obj_label
        target_out['masks'] = reshape_obj_masks
        target_out['aff_map'] = aff_map
        
        return img, target_out

class Pad(object):
    """Pad the image & masks & segmentation map.
    Args:
        size_divisor (int, optional): The divisor of padded size.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self, size_divisor):
        self.size_divisor = size_divisor

    def pad_img(self, img):
        """Pad images according to ``self.divisor``."""
        padded_img = mmcv.impad_to_multiple(img, self.size_divisor, pad_val = 0)
        return padded_img

    def pad_masks(self, img, target):
        """Pad masks according to ``results['pad_shape']``."""
        obj_masks = target['masks']
        pad_shape = img.shape[:2]
        padded_masks = []
        for i in range(len(obj_masks)):
            padded_masks.append(mmcv.impad(obj_masks[i], shape = pad_shape, pad_val = 0))
        target['masks'] = padded_masks

    def pad_seg(self, img, aff_map):
        """Pad semantic segmentation map according to ``results['pad_shape']``."""
        pad_shape = img.shape[:2]
        aff_map_out = mmcv.impad(aff_map, shape = pad_shape, pad_val = 0)
        return aff_map_out

    def __call__(self, img, target):
        img_out = self.pad_img(img)
        self.pad_masks(img_out, target)
        aff_map = target['aff_map'].copy()
        target['aff_map'] = self.pad_seg(img_out, aff_map)
        return img_out, target

class Pad_img(object):
    def __init__(self, size_divisor):
        self.size_divisor = size_divisor

    def __call__(self, img):
        return mmcv.impad_to_multiple(img, self.size_divisor, pad_val = 0)

class RandomFlip(object):
    'Makes an horizontal flip randomly'
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img, target):
        obj_bbox, obj_label, obj_masks, aff_map = target['boxes'], target['labels'], target['masks'], target['aff_map']
        h, w = img.shape[:2]
        rdn = random.uniform(0, 1)
        target_out = {}
        target_out['labels'] = obj_label
        
        if rdn > self.prob:
            flip = True
            flip_img = img.copy()
            img = np.fliplr(flip_img)
            initial_bbox = obj_bbox.copy()
            obj_bbox[:, 0] = w - initial_bbox[:, 2]
            obj_bbox[:, 2] = w - initial_bbox[:, 0]
            flip_obj_masks = []
            for i in range(len(obj_masks)):
                flip_obj_mask_i = np.fliplr(obj_masks[i])
                flip_obj_masks.append(flip_obj_mask_i)
            target_out['boxes'] = obj_bbox
            target_out['masks'] = flip_obj_masks
            target_out['aff_map'] = np.fliplr(aff_map)
            
        else:
            flip = False
            target_out['boxes'] = obj_bbox
            target_out['masks'] = obj_masks
            target_out['aff_map'] = target['aff_map']

        return img, target_out


class Input_noise(object):
    def __init__(self, margin):
        self.h_max = margin[0]
        self.w_max = margin[1]

    def __call__(self, img, target):
        new_img = np.zeros_like(img)
        top = np.random.randint(0, self.h_max)
        wide = np.random.randint(0, self.w_max)
        new_img[top:new_img.shape[0], wide:new_img.shape[1], :] = img[0: new_img.shape[0] - top, 0: new_img.shape[1] - wide, :]
        return new_img, target

class RandomFlip_img(object):
    'Makes an horizontal flip randomly'
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img):
        rdn = random.uniform(0, 1)
        if rdn > self.prob:
            flip_img = img.copy()
            img = np.fliplr(flip_img)
        return img

class Normalize(object):
    'Normalize the img to 103.530, 116.280, 123.675 values'
    def __init__(self, mean_values):
        self.mean_values = mean_values
        self.std_values = [0.229, 0.224, 0.225]
    
    def __call__(self, img, target):
        norm_img = img.copy()
        norm_img[:, :, 0] = (img[:, :, 0] - self.mean_values[0]) / self.std_values[0]
        norm_img[:, :, 1] = (img[:, :, 1] - self.mean_values[1]) / self.std_values[1]
        norm_img[:, :, 2] = (img[:, :, 2] - self.mean_values[2]) / self.std_values[2]
        return norm_img, target

class Normalize_img(object):
    'Normalize the img to 103.530, 116.280, 123.675 values'
    def __init__(self, mean_values):
        self.mean_values = mean_values
        self.std_values = [0.229, 0.224, 0.225]
    
    def __call__(self, img):
        norm_img = img.copy()
        norm_img[:, :, 0] = (img[:, :, 0] - self.mean_values[0]) / self.std_values[0]
        norm_img[:, :, 1] = (img[:, :, 1] - self.mean_values[1]) / self.std_values[1]
        norm_img[:, :, 2] = (img[:, :, 2] - self.mean_values[2]) / self.std_values[2]
        return norm_img

class RandomCrop(object):
    'Crop randomly the image in a sample'
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, target):
        obj_bbox, obj_label, obj_masks, aff_map = target['boxes'], target['labels'], target['masks'], target['aff_map']
        
        #Crop the image
        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        old_img = img
        img = img[top: top + new_h, left: left + new_w]
        #depth = depth[top: top + new_h, left: left + new_w]
        aff_map = aff_map[left: left + new_w, top: top + new_h]
        reshape_obj_masks = []
        for i in range(len(obj_masks)):
            reshape_obj_masks.append(obj_masks[i][top: top + new_h, left: left + new_w])
        obj_bbox = (obj_bbox - [left, top, left, top]).astype(int)
        
        #aff_map_show = aff_map.squeeze().flatten().astype(int)
        #count_array = np.bincount(aff_map_show)
        #print('Distribution of affordances after the cropping',count_array)
        #print(aff_map.shape, aff_map.dtype)
        
        target_out = {}
        target_out['boxes'] = obj_bbox
        target_out['labels'] = obj_label
        target_out['masks'] = reshape_obj_masks
        target_out['aff_map'] = aff_map

        return img, target_out

class RandomCrop_img(object):
    'Crop randomly the image in a sample'
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, target):
        #Crop the image
        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        return img

class ToTensor(object):
    'Convert all to tensors'    
    def __call__(self, img, target):
        aff_bbox, aff_label, aff_masks, aff_map = target['boxes'], target['labels'], target['masks'], target['aff_map']
        img = img.transpose((2, 0, 1))
        
        img = torch.from_numpy(img.copy()).type(torch.float) / 255
        #depth = depth.transpose((2, 0, 1))
        #depth = torch.from_numpy(depth).type(torch.float)
        #'aff_map': torch.from_numpy(aff_map).type(torch.int),
        target_out = {}
        target_out['boxes'] = torch.from_numpy(aff_bbox).type(torch.float) #boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        target_out['labels'] = aff_label.type(torch.int64) #labels (``Int64Tensor[N]``): the class label for each ground-truth box
        target_out['masks'] = torch.Tensor(np.array(aff_masks)).type(torch.uint8)
        target_out['aff_map'] = torch.from_numpy(aff_map.copy()).type(torch.int)
        return img, target_out

class ToTensor_img(object):
    'Convert all to tensors'    
    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img.copy()).type(torch.float)