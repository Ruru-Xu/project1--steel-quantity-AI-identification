import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/faster_rcnn_x101_32x4d_fpn_1x.py')
cfg.model.pretrained = '/home/imc/XR/models/chenxu/mmdetection/pretrained/faster_rcnn_x101_32x4d_fpn_1x/faster_rcnn_x101_32x4d_fpn_1x.pth'

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
#_ = load_checkpoint(model, 'model/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
img = mmcv.imread('2.jpg')
result = inference_detector(model, img, cfg)
show_result(img, result)

# test a list of images
# imgs = ['1.jpg', '2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     print(i, imgs[i])
#     show_result(imgs[i], result)
