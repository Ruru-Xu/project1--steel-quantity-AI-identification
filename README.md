# project1--steel quantity AI identification

I am currently participating in a competition.

**In fact, we are a team, our team have five members, this competition is the result of our efforts together. Regardless of the outcome whether we'ill get the prize. It is also a reward for me to know a few well-behaved friends.**

The background of the competition : 

*At the construction site, the acceptance personnel shall manually count the number of rebars loaded on the trucks entering the site, and the trucks can unload only after the quantity is confirmed. Manual counting is currently used at the site,*
*The process is tedious, labor-intensive and slow (usually it takes several hours to count a truckload of rebars). In order to solve the above problems, we hope to accomplish this task intelligently and efficiently through mobile phone photographing-> target detection and counting-> manual little errors modification*

I have tried：

- It took me a long time on YOLOv3，It’s hard to improve after the accuracy is 0.95.
- Trained the Faster RCNN model with Tensorflow API，But the situation is not optimistic
- I tried to fuse the result of YOLOv3 with the result of the Faster RCNN. The purpose is that I want the Faster RCNN to make up for the problem of YOLOv3 missed detection.The final score is 0.97, which can't be higher.
- After discussing it with my classmates，We think that YOLOv3 is not sensitive to small targets, the model in Tensorflow API is not optimized, we can't waste any more time.So we moved the target to mmdetection. Currently our score has reached 0.986669

------

## Let's start !!!!

**1-mmdetection_install-training.md**

```
These .py files involved: pascal_voc_xml2json.py,  test_coco.py,  test_image.py,  inference.py,  show_bboxes.py
```

