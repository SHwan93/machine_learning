# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:26:47 2018

@author: HWAN
"""
import tensorflow
import matplotlib.pyplot as plt
import scipy.cluster.vq as sccv
import numpy as np
import PIL
#import sh3d

img = PIL.Image.open('000501.jpg')
img_pix = np.array(img, dtype=np.float32) #1906x2560 

all_pix = [pixel for row in img_pix for pixel in row] # 위 부터 아래까지 행 기준으로 모든 픽셀 알아내기
pix = np.array(all_pix)
 
r = pix[:,0] 
g = pix[:,1]
b = pix[:,2]
 
plt.imshow(img_pix)
#sh3d.my3d(r,g,b)


#data = np.vstack((np.random.rand(150,2) + np.array([.5,.5]),np.random.rand(150,2)))
#
#centroids , _ = sccv.kmeans(all_pix,2)
#
#idx, _ = sccv.vq(all_pix ,centroids)

#plt.plot(data[idx==0,0],data[idx==0,1],'ob',data[idx==1,0],data[idx==1,1],'or')
#plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
#plt.show()