import cv2
import numpy as np
import random


def group_dynamic_blend(sources, targets, masks):
    imgs_blended = []
    seed = np.random.randint(2147483647)
    for source, target, mask in zip(sources, targets, masks):
        img_blended = dynamic_blend(source, target, mask, seed)
        imgs_blended.append(img_blended)
    return imgs_blended
    

def dynamic_blend(source,target,mask, seed):
	mask_blured = get_blend_mask(mask, seed)
	blend_list=[0.25,0.5,0.75,1,1,1]
	random.seed(seed)
	np.random.seed(seed)
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured*=blend_ratio
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended

def get_blend_mask(mask, seed):
	H,W=mask.shape
	random.seed(seed)
	np.random.seed(seed)
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))