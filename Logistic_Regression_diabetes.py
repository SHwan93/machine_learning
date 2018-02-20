# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:41:45 2018

@author: HWAN
"""
import matplotlib.pyplot as plt
import numpy as np
xy=np.loadtxt('data-03-diabetes1.csv',delimiter=',',dtype=np.float32)
x=xy[:,0:-1]
y=xy[:,[-1]]

a=np.random.random([8,1])#8
b=np.random.random(1)#-16

def sigmoid(a,b,x):
    
    return 1/(1+np.exp(-(np.dot(x,a)+b)))  # 다중이면 웬지 여기를 dot 으로 바꿔주면 될것같은데 ,a,b 배열 공간도 할당 해줘야함

def cross_entropy(f,a,b,x,y):
    
    loss = np.sum((-y*np.log(f(a,b,x)))-(1-y)*np.log(1-f(a,b,x)))/x.shape[1] # 지금은 0이지만 2차원부터는 1로
    
    return loss

def diff_a(f,hy,a,b,x,y):
    
    return ((f(hy,a+h,b,x,y)-f(hy,a-h,b,x,y))/(2*h))

def diff_b(f,hy,a,b,x,y):
    
    return ((f(hy,a,b+h,x,y)-f(hy,a,b-h,x,y))/(2*h))

lr = 0.001
h = 1e-4
tmp=[]
for step in range(3000):
    
    b = b-lr*diff_b(cross_entropy,sigmoid,a,b,x,y)

for step in range(3000):
      
    a = a-lr*diff_a(cross_entropy,sigmoid,a,b,x,y)   
    

acc=[]
for i in sigmoid(a,b,x):
    
    if(i > 0.5):
        acc.append(1)
        #print('1')
    else:
        acc.append(0)
        #print('2')
acc = np.expand_dims(acc,1)
pre = (np.equal(acc,y))
accuracy = np.sum(pre.astype(float))/pre.shape[0]
print('정확도 : %f' %accuracy)
c = cross_entropy(sigmoid,a,b,x,y)
print('손실',c)
#''' ROC곡선 만들기 '''
#TP_count=0 ;FP_count=0;TN_count=0;FN_count = 0
#
#for i,j in zip(y,acc):
#    
#    if( i == 1 and j ==1): # TP 암인데 암이라고 판정한거
#        TP_count += 1        
#    elif( i == 0 and j == 1): # FP  암이아닌데 암이라고 판정한거
#        FP_count += 1
#    elif( i == 1 and j == 0): #TN 암이 맞는데 아니라고 판정
#        TN_count +=1
#    elif( i==0 and j == 0): # FN 암이 아닌데 아니라고 판정 
#        FN_count +=1
#        
## 민감도 = 진양성률 (진양성/진음성+진양성)
## 1-특이도 = 위양성률 (위양성 /위양선+위음성)
#        
#TPN = TP_count/(TP_count+TN_count)
#FPN = FP_count/(FP_count+FN_count)
#
#plt.scatter(FPN,TPN)
#plt.show()