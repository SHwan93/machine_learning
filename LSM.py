# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:04:44 2018

@author: HWAN
"""

#http://darkpgmr.tistory.com/56?category=761008

import numpy as np
import matplotlib.pyplot as plt

def dataset(num):
    data_x = np.arange(0,1,1/num)
    data_y = [np.sin(2*np.pi*i)+ np.random.normal(scale=0.3) for i in data_x]
    
    real_x = np.arange(0,1,1/num)
    real_y = [np.sin(2*np.pi*i) for i in real_x]
    
    
    return data_x,data_y,real_x ,real_y
    
def test(d_x , r_y , dimension ):
    tmp=[]
    d_x = np.array(d_x)
    
    for i in range(1,dimension+1):
        d_x = d_x**i
        tmp += list(d_x)
    tmp = np.array(tmp)
    tmp = np.expand_dims(tmp,1)
    k = tmp.reshape([dimension,d_x.size])
    kk = k.T
    kk = np.concatenate((kk,np.ones([d_x.size,1])), axis=1)
    inv_kk = np.linalg.pinv(kk)
    
    r_y = np.expand_dims(r_y, axis=1)
    w = np.dot(inv_kk , r_y)
  
    
    return  w,kk     
   
   
d_x , d_y , r_x ,r_y = dataset(1000)
train_w ,train_x = test(d_x , d_y ,3)

train_y = np.dot(train_x,train_w)
error = np.sum(np.square((np.expand_dims(d_y,1)-train_y)), axis=0)

plt.xlim(-0.1,1.1)
plt.ylim(-1.5,2)
plt.plot(r_x, r_y ,'r--' ,label='real') # 진짜 함수로 그린거
plt.scatter(d_x,d_y)
plt.plot(d_x,train_y, 'b', label='prediction' ) # 예측 함수로 그린거
plt.title('error = %f' %error)
plt.legend(loc=1)
 
plt.plot() 
plt.show()

# 검증
test_d_x ,test_d_y , test_r_x, test_r_y = dataset(1000)
_ ,test_x = test(test_d_x , test_r_y ,3)
test_y = np.dot(test_x,train_w)

error = np.sum(np.square((np.expand_dims(test_d_y,1)-test_y)), axis=0)

plt.xlim(-0.1,1.1)
plt.ylim(-1.5,2)
plt.plot(test_r_x, test_r_y ,'r--' )    
plt.scatter(test_d_x,test_d_y)
plt.plot(test_d_x,test_y, 'b')
plt.legend(loc=1)
plt.title('error = %f' %error)
plt.show()

# 검증
e_t=[]
for i in range(1,21):    
    train_w ,train_x = test(d_x , d_y ,i)
    
    train_y = np.dot(train_x,train_w)
    error = np.sum(np.square((np.expand_dims(d_y,1)-train_y)), axis=0)
    e_t.append(error)
plt.plot(np.arange(1,21) , e_t)
plt.show()
