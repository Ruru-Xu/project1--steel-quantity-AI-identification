#-*- coding:utf-8 -*-
import os 
import time
import mmcv
import cv2
import shutil
import skimage.io as io
import numpy as np 
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector

_BLUE = (0, 0, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


"""
这个脚本可视化检测结果
"""

class SHOW_BBOXES(object):
	"""docstring for SHOW_BBOXES"""
	def __init__(self, img_path,cfg_path,model_path,save_img_path):
		super(SHOW_BBOXES, self).__init__()
		self.img_path = img_path
		self.cfg_path = cfg_path
		self.model_path = model_path
		self.save_img_path = save_img_path
		self.images = os.listdir(self.img_path)
		self.score_thr = 0.85#得分阈值自己改

	def inference_reult(self):
		"""
		return 
		{
			"img_name":{"bboxes":[[x11,y11,w1,h1],...,[xn1,yn1,wn,hn]],"scores":[s1,s2,...,sn]},
			...
		}
		"""
		cfg = mmcv.Config.fromfile(self.cfg_path)
		cfg.model.pretrained = None

		# construct the model and load checkpoint
		model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
		_ = load_checkpoint(model, self.model_path)
		total_imgs = len(self.images)
		infer_dict = {}
		for num,img in enumerate(self.images):
			end = time.time()
			img_name = img
			if img_name not in infer_dict.keys():
				infer_dict[img_name] = {}
				infer_dict[img_name]["bboxes"] = []
				infer_dict[img_name]["scores"] = []

				img = mmcv.imread(self.img_path + img)
				result = inference_detector(model, img, cfg)
				scores = result[0][:,4]
				bboxes = result[0][:,0:4].tolist()
				for i,s in enumerate(scores):
					if s >= self.score_thr:
						infer_dict[img_name]["scores"].append(s)
						infer_dict[img_name]["bboxes"].append(bboxes[i])

				# bboxes = result[0][:,0:4].tolist()
				# scores = result[0][:,4]
				# infer_dict[img_name]["bboxes"].append(bboxes)
				# infer_dict[img_name]["scores"].append(scores)
				print("the [%d/%d] cost time :%s" %(num,total_imgs,time.time() - end))
			else:
				print("image exists ...")
		return infer_dict

	def vis_bbox(self,img, bbox, thick=2):
		"""Visualizes a bounding box."""
		img = img.astype(np.uint8)
		(x0, y0, x1, y1) = bbox
		x1, y1 = int(x1), int(y1)
		x0, y0 = int(x0), int(y0)
		cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
		return img

	def vis_class(self,img, pos, class_str,score, font_scale=1):
		"""Visualizes the class."""
		img = img.astype(np.uint8)
		x0, y0 = int(pos[0]), int(pos[1])
		# Compute text size.
		txt = class_str + "  " + str(score)
		font = cv2.FONT_HERSHEY_SIMPLEX
		((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
		# Place text background.
		back_tl = x0, y0 - int(1.3 * txt_h)
		back_br = x0 + txt_w, y0
		cv2.rectangle(img, back_tl, back_br, _RED, -1)
		# Show text.
		txt_tl = x0, y0 - int(0.3 * txt_h)
		cv2.putText(img, txt, txt_tl, font, font_scale, _BLUE, lineType=cv2.LINE_AA)
		return img


	def save_draw_imgs(self):
		if os.path.exists(self.save_img_path):
			shutil.rmtree(self.save_img_path)
			os.makedirs(self.save_img_path)
		else:
			os.makedirs(self.save_img_path)
		result_dict = self.inference_reult()
		for img,dets in result_dict.items():
			img_path = self.img_path + img 
			save_img = os.path.join(self.save_img_path,img)
			my_img = io.imread(img_path)
			bboxes = dets['bboxes']
			scores = dets['scores']
			for i,bbox in enumerate(bboxes):
				score = round(scores[i],2)
				my_img = self.vis_bbox(my_img,bbox)
				#my_img = self.vis_class(my_img,(bbox[0],bbox[1] -1),"g",score)
			b,g,r = cv2.split(my_img)
			my_img = cv2.merge([r,g,b])
			cv2.imwrite(save_img,my_img)


if __name__ == "__main__":
	#这几个路径自己对应
	img_prefix_path = "/home/my_dataset/gangjin-mm/coco/test2017/"
	cfg_path = "configs/faster_rcnn_r101_fpn_1x.py"
	model_path = "/home/mmdetection/gangjin/0220/latest.pth"
	save_img_path = "/home/vis_boxes/mmdetection/gangjin"

	my_bboxes = SHOW_BBOXES(img_prefix_path,cfg_path,model_path,save_img_path)
	my_bboxes.save_draw_imgs()
