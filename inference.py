#-*- coding:utf-8 -*-
import os 
import time
import mmcv
import pandas as pd 
import numpy as np 
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector


"""
这个脚本是直接生产提交文件
"""
#这三个路径自己对应
img_prefix_path = "/home/my_dataset/gangjin-mm/coco/test2017/"
cfg_path = "configs/faster_rcnn_r101_fpn_1x.py"
model_path = "/home/mmdetection/gangjin/0220/latest.pth"

score_thr = 0.75 #自己跳大调小过滤

cfg = mmcv.Config.fromfile(cfg_path)
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, model_path)

submit_list = []
# test a list of images
imgs = os.listdir(img_prefix_path)
for i,img in enumerate(imgs):
	end = time.time()
	img_name = img
	img = mmcv.imread(img_prefix_path + img)
	result = inference_detector(model, img, cfg)
	bboxes = result[0]
	for box in bboxes:
		line_data = []
		score = box[4]
		if score <= score_thr:
			#过滤掉分数很低的图片
			continue
		box_str = str(round(box[0]))+" "+str(round(box[1]))+" "+str(round(box[2]))+" "+str(round(box[3]))
		line_data.append(img_name)
		line_data.append(box_str)
		submit_list.append(line_data)
	print("the {}th img, cost time: {:.4f}".format(i,time.time() - end))
print("one box format:",box)
df = pd.DataFrame(np.array(submit_list))
#提交csv的保存路径
df.to_csv("/home/vis_boxes/mmdetection/mysubmit.csv",encoding="utf-8",header=False,index=False)
