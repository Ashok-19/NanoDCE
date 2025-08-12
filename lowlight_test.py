import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
#import student
import model
import model_student
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 12
	data_lowlight = Image.open(image_path)
	
	
	
	data_lowlight = (np.asarray(data_lowlight)/255.0)
	
	
	data_lowlight = torch.from_numpy(data_lowlight).float()
	
	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)
	
	#DCE_net = student.SmallZeroDCE(scale_factor).cuda()
	DCE_net = model_student.enhance_net_nopool_student(scale_factor).cuda()
	#DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	DCE_net.load_state_dict(torch.load('snapshots_Student_KD_both/Student_Final.pth'))
	#DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth'))
	
	# --- robust checkpoint loading (replace your single load_state_dict call) ---
	'''
	#--------------------------------------------------------------------------------------#
	
	ckpt_path = 'distill_nf8_checkpoints/student_nf8_2_best.pth'
	ck = torch.load(ckpt_path, map_location='cpu')
	
	# Unwrap common container formats
	if isinstance(ck, dict):
		if 'student_state' in ck:
			state = ck['student_state']
		elif 'state_dict' in ck:
			state = ck['state_dict']
		else:
			# maybe it's already a raw state_dict
			# detect whether dict keys look like param names or wrapper keys
			some_keys = list(ck.keys())[:10]
			if any(k.startswith('e_conv') or k.startswith('module.') for k in some_keys):
				state = ck
			else:
				raise RuntimeError(f"Unrecognized checkpoint format. Keys: {list(ck.keys())[:10]}")
	else:
		state = ck
	
	# If saved from DataParallel it may have 'module.' prefix on keys — remove if present
	new_state = {}
	for k,v in state.items():
		new_key = k
		if k.startswith('module.'):
			new_key = k[len('module.'):]
		new_state[new_key] = v
	
	# Ensure your model was created with the same architecture as the saved student
	# (make sure number_f=8 for student)
	# For example:
	# DCE_net = model.enhance_net_nopool(scale_factor=1, number_f=8).cuda()
	
	DCE_net.load_state_dict(new_state)
	print(f"Loaded student state from {ckpt_path}")
	# --------------------------------------------------------------------------'''
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)
	
	end_time = (time.time() - start)
	
	print(end_time)
	#image_path = image_path.replace('test_data','result_Zero_DCE++')
	image_path = image_path.replace('test_data','result_student')
	
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	with torch.no_grad():

		filePath = 'data/test_data/'	
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
			
				print(image)
				sum_time = sum_time + lowlight(image)

		print(sum_time)
		

