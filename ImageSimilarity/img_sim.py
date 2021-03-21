################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import argparse
import os
import util
import cv2
import csv
import metrics
#from skimage.metrics import structural_similarity


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/test01/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/test01/ex_dir1')
parser.add_argument('-o','--dir_out', type=str, default='./result/test01/')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

field_names = ['Patient_ID', 'MSE', 'AE', 'PSNR', 'SSIM', 'LPIPS']

#to_write = [['Patient_ID', 'MSE', 'AE', 'PSNR', 'SSIM', 'LPIPS']]
#
#
# if not os.path.exists(opt.dir_out):
# 	os.mkdir(opt.dir_out)
# util.save_to_file(opt.dir_out + '/result.csv', to_write)


with open(opt.dir_out + '/result.csv', mode='w', newline='') as file:
	writer = csv.DictWriter(file, fieldnames=field_names)
	writer.writeheader()
	# crawl directories
	# f = open(opt.out,'w')
	files = os.listdir(opt.dir0)


	for file in files:
		if(os.path.exists(os.path.join(opt.dir1,file))):

			f_dict = {fn: '-' for fn in field_names}
			f_dict['Patient_ID'] = file

			# Load images
			img0 = util.load_image(os.path.join(opt.dir0,file))
			img1 = util.load_image(os.path.join(opt.dir1,file))

			img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
			img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

			img0_tensor = util.np2tensor(util.normalize_img(img0))
			img1_tensor = util.np2tensor(util.normalize_img(img1))

			if(opt.use_gpu):
				img0_tensor = img0_tensor.cuda()
				img1_tensor = img1_tensor.cuda()

			# Compute distance
			#dists['LPIPS'] = loss_fn.forward(img0_tensor, img1_tensor)
			metric = metrics.MSE()
			f_dict['MSE'] = metric(img0_tensor, img1_tensor).item()
			metric = metrics.PSNR()
			f_dict['PSNR'] = metric(img0_tensor, img1_tensor).item()
			metric = metrics.SSIM()
			f_dict['SSIM'] = metric(img0_tensor, img1_tensor).item()
			metric = metrics.LPIPS(opt.use_gpu)
			f_dict['LPIPS'] = metric(img0_tensor, img1_tensor).item()
			metric = metrics.AE()
			f_dict['AE'] = metric(img0_tensor, img1_tensor).item()

			print(file)
			print('MSE : {:.3f}'.format(f_dict['MSE']))
			print('AE : {:.3f}'.format(f_dict['AE']))
			print('PSNR : {:.3f}'.format(f_dict['PSNR']))
			print('SSIM : {:.3f}'.format(f_dict['SSIM']))
			print('LPIPS : {:.3f}'.format(f_dict['LPIPS']))
			print('-'*8)
			writer.writerow(f_dict)

# 		to_write.append([file, '{:.4f}'.format(dists['MSE']), dists['AE'], dists['PSNR'], dists['SSIM'], dists['LPIPS']])
# 		to_write.append([file, '{:.4f}'.format(dists['MSE']), dists['AE'], dists['PSNR'], dists['SSIM'], dists['LPIPS']])
#
# #
# f.close()
