#-*- utf-8 -*-
from PIL import Image
import numpy as np
import json
import os 

"""
生成测试图片的json文件，coco格式
只有 categories 和 images字段
"""

class COCO_TEST(object):
	"""docstring for COCO_TEST"""
	def __init__(self,train_imgs,save_json_path='./instances_test2014.json'):
		super(COCO_TEST, self).__init__()
		self.train_imgs = train_imgs
		self.save_json_path = save_json_path
		self.test_images = os.listdir(train_imgs)
		self.images = []	#coco 中的images字段
		self.categories = [self.categoie()]	#coco 中的categories字段
		self.label = ["g"]		#surprisely数据集中只有一类 -->person
		self.height = 0		#coco 中images字段列表中中每张图片的 高度
		self.width = 0		#coco 中images字段列表中中每张图片的 宽度

		# self.save_json()


	def data_transfer(self):
		for num,img in enumerate(self.test_images):
			name = img.split('.')[0]
			# ann_name = self.ann_prefix + json_file
			img_name = self.train_imgs + img
			print(img_name)
			self.images.append(self.image(img_name,num))


	def image(self,img_path,num):
		'''
		img_path: the path of one annotated image
		num: the current image id, unique label
		return a dict about the image info

		这个函数对用json文件中的images字段列表中的一个字段元素，
		该字典包含了训练需要用的信息
		'''
		image = {}
		img = Image.open(img_path)
		w,h = img.size
		image['height'] = h
		image['width'] = w
		img = None
		image['id'] = num + 1
		image['file_name'] = img_path.split('/')[-1]

		self.height = h
		self.width = w

		return image

	def categoie(self):
		'''
		这个函数对应coco 中的 categories字段list
		'''
		categorie = {}
		categorie['supercategory'] = 'g'	#本数据集只有一类人，所以父类和子类都是person
		categorie['id'] = 1	#默认person标签为1
		categorie['name'] = 'g'	#本数据集只有一类人，所以父类和子类都是person

		return categorie

	def data2coco(self):
		"""
		这个函数最终生成需要保存的json文件，这个json文件可用来训练
		其中只需要用到三个字段
		"""
		data_coco = {}
		data_coco['images'] = self.images
		data_coco['categories'] = self.categories

		return data_coco

	def save_json(self):
		self.data_transfer()
		self.data_coco = self.data2coco()
		json.dump(self.data_coco,open(self.save_json_path,'w'),indent=4)

if __name__ == "__main__":
	test_imgs_path = "C:/Users/xuru/Desktop/test_dataset/"
	save_json_name = "C:/Users/xuru/Desktop/0110/instances_test2014.json"

	test_img = COCO_TEST(test_imgs_path,save_json_name)
	#generate json file
	test_img.save_json()