import numpy as np

def material_from_gt_label(gt_labelmap):
	""" Merges several classes. """

	if not len(gt_labelmap.shape) == 2:
	    gt_labelmap = gt_labelmap[:,:,:3]
	    gt_labelmap = gt_labelmap[:,:,2]

	h,w = gt_labelmap.shape
	shader_map = np.zeros((h, w, 12), dtype=np.float32)
	
	shader_map[:,:,0] = (gt_labelmap == 23).astype(np.float32) # sky
	shader_map[:,:,1] = (np.isin(gt_labelmap, [6, 7, 8, 9, 10])).astype(np.float32) # road / static / sidewalk
	shader_map[:,:,2] = (np.isin(gt_labelmap, [26,27,28,29,30,31,32,33])).astype(np.float32) # vehicle
	shader_map[:,:,3] = (gt_labelmap == 22).astype(np.float32) # terrain
	shader_map[:,:,4] = (gt_labelmap == 21).astype(np.float32) # vegetation
	shader_map[:,:,5] = (np.isin(gt_labelmap, [24,25])).astype(np.float32) # person
	shader_map[:,:,6] = (np.isin(gt_labelmap, [17,18])).astype(np.float32) # infrastructure
	shader_map[:,:,7] = (gt_labelmap == 19).astype(np.float32) # traffic light
	shader_map[:,:,8] = (gt_labelmap == 20).astype(np.float32) # traffic sign
	shader_map[:,:,9] = (gt_labelmap == 1).astype(np.float32) # ego vehicle
	shader_map[:,:,10] = (np.isin(gt_labelmap, [4, 11, 12, 13, 14, 15, 16])).astype(np.float32) # building
	shader_map[:,:,11] = (np.isin(gt_labelmap, [0,5])).astype(np.float32) # unlabeled
	
	#import matplotlib.pyplot as plt

	#plt.imshow(shader_map[:,:,1])
	#plt.show()
	
	
	return shader_map
