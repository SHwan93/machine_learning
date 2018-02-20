# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:28:18 2018

@author: HWAN
"""

import sklearn.mixture as skm
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)
''' A GaussianMixture.fit method is provided that learns a Gaussian Mixture Model from train data. 
Given test data, it can assign to each sample 
the Gaussian it mostly probably belong to using the GaussianMixture.predict method.

GaussianMixture.fit 메서드는 학습 데이터에서 가우스 혼합 모델을 학습합니다.
테스트 데이터가 주어지면 각 샘플에 할당 할 수 있습니다.
Gaussian은 GaussianMixture.predict 메소드를 사용하여 대부분 속해 있습니다.

GaussianMixture에는 구형, 대각선, 묶음 또는 전체 공분산과 같은 차이 클래스의 공분산을 제한하기 위해 다양한 옵션이 있습니다.


우선 GMM을 알기 위해서는 Mixture Model을 알아야합니다. Mixture Model 전체 분포에서 하위 분포가 존재한다고 보는 모델입니다. 
즉, 데이터가 모수를 갖는 여러개의 분포로부터 생성되었다고 가정하는 모델입니다. 
책에서는 보통 이를 두고, "Unsupervised Learning의 모수적 접근법"이라고 합니다.  
이 중에서 가장 많이 사용되는 가우시안 믹스쳐 모델(GMM)은 데이터가 K개의 정규분포로부터 생성되었다고 보는 모델입니다.

'full' (each component has its own general covariance matrix),
'tied' (all components share the same general covariance matrix),
'diag' (each component has its own diagonal covariance matrix),
'spherical' (each component has its own single variance).

score_samples(X)	Compute the weighted log probabilities for each sample.
score(X[, y])	Compute the per-sample average log-likelihood of the given data X.
predict(X)	Predict the labels for the data samples in X using trained model.
fit(X[, y])	Estimate model parameters with the EM algorithm.
'''

        
data1 = [np.random.normal(-0.5,0.2) for i in range(2000)]
data2 = [np.random.normal(-0.1,0.07) for i in range(5000)]
data3 = [np.random.normal(0.2,0.13) for i in range(10000)]
data = data1+data2+data3
data = np.expand_dims(data , 1)
print(len(data1),len(data2),len(data3))

plt.subplot(311)
plt.title('three data come from normal distribution ')
plt.hist(data1)
plt.hist(data2,alpha=0.7)
plt.hist(data3,alpha=0.7)

plt.subplot(313)
plt.title("one data that we'll look")
plt.hist(data1,color='r')
plt.hist(data2,color='r')
plt.hist(data3,color='r')
plt.show()

''' 우리가 알아야할건 총 9개 ( 세가지 분포의 평균, 분산 , 세가지 분포의 weight)

실제 모수 

Weight1 = 2000/(2000+5000+10000) = 0.117
Weight2 = 5000/(2000+5000+10000) = 0.29
Weight3 = 10000/(2000+5000+10000) = 0.588
평균1 = -0.5
평균2 = -0.1
평균3=  0.2
표준편차1 = 0.2
표준편차2 = 0.07
표준편차3 = 0.13

GMM에서의 모수는 두 가지 종류가 있습니다.
첫 번째는 3가지 정규분포 중 확률적으로 어디에서 속해있는가를 나타내는 Weight 값, 
두 번째, 각각의 정규분포의 모수(평균, 분산)입니다. 첫 번째 종류의 모수를 잠재변수 라고 부르며, 
잠재변수가 포함된 모델은 Mixture Model에서의 모수 추정은 MLE(maximum likelihood estimation)으로 구할 수 없기 때문에 
EM(Expectation Maximazation)이라고 부르는 알고리즘을 통해 iterative하게 구하게 됩니다.
(왜냐하면 잠재변수가 포함되었기 때문에 likelihood function을 미분할 수가 없기 때문입니다.

'''
gmm = skm.GaussianMixture(n_components=3, covariance_type='full' )
gmm.fit(data)
gg = gmm.predict(data) 
g = gmm.score_samples(data) #로그 값 가중치


        

