# A TensorFlow implementation of TCDCN

#사용법 
#아래 Train데이터를 받고 (현재위치에 압축을 풀어준다. 코드상에서의 경로설정 때문에)
#Download Data uRL : https://drive.google.com/open?id=10FAFInq3u72V3s2KDnwZFAe4f1YJVR4c
#MainNet.py 파일과 테스트하고 영상으로 볼 수 있는 testimage.py 파일을 받는다.
#img_size = 40 크기로 학습할 이미지 크기를 설정할 수 있지만 40이 

It is a TensorFlow implementation of《[Facial Landmark Detection by Deep Multi-task Learning](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)》

>Instead of treating the detection task as a single and independent problem, we investigate the possibility of improving detection robustness through multi-task learning.

## Network structure
![structure](http://img.blog.csdn.net/20161003235731201)

## Libraries
You need to install OpenCV3 for image processing.

## Chinese version
You can read my blog for more information:
>http://blog.csdn.net/tinyzhao/article/details/52730553
