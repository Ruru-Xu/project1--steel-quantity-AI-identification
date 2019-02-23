Not so much nonsense, follow my steps

- ## install mmdetection

  Create a virtual environment and operate in a virtual environment

  ```
  conda create -n Mmdetection python=3.6
  conda activate Mmdetection
  ```

  ![1550925579972](img/1550925579972.png)

  Install some necessary packages

```
unzip mmdetection-master.zip #I have downloaded it in advance, directly extract it.
cd mmdetection
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision #Install pytorch through Tsinghua source
pip install decorator cloudpickle>=0.2.1 dask[array]>=1.0.0 matplotlib>=2.0.0 networkx>=1.8 scipy>=0.17.0 bleach python-dateutil>=2.1 decorator #Install these required packages according to the red code hints
pip install cython
./compile.sh  #These two lines of code, install cython
python setup.py install #Install mmdetection
```

![1550925604767](img/1550925604767.png)

- ## Making our own datasets:  Coco format

  annotations----------Store the .json files

  The other three folders store test sets, training sets, and validation sets.

  ![1550925654468](img/1550925654468.png)

  For information on how to make a coco format dataset, please refer to my three .py files.

```
pascal_voc_xml2json.py

test_coco.py

test_image.py
```

- ## Training

  training code

  ```
  python tools/train.py configs/faster_rcnn_x101_32x4d_fpn_1x.py --gpu 2
  ```

  ![1550920264271](img/1550920264271.png)

Model is placed in the configs folder

![1550920340426](img/1550920340426.png)

There are some parameters in the .py file that need to be changed, you can obviously see that like this

![1550920383163](img/1550920383163.png)

You can fine tune the scale in the train code block to training

![1550920548009](img/1550920548009.png)

- ## Test

Test code

```
python tools/test.py configs/cascade_rcnn_r101_fpn_1x.py gangjin/cascade_rcnn_r101_fpn_1x_0222/latest.pth --gpus 2 --out gangjin/cascade_rcnn_r101_fpn_1x/result.pkl --eval bbox
```

![1550920691202](img/1550920691202.png)

You can fine tune the scale in the test code block to test

```
vim cascade_rcnn_r101_fpn_1x.py
```

![1550920936785](img/1550920936785.png)

There are some parameters in the inference.py file that need to be changed,like threshold

![1550921594360](img/1550921594360.png)

- ## visualization

  refer to my show_bboxes.py file

![1550921876643](img/1550921876643.png)

![1550921905319](img/1550921905319.png)