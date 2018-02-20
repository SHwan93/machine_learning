# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:16:21 2018

@author: HWAN
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

data=[]
spam_all_text = []
all_text=[]
spam_cnt=0
ham_cnt=0

f=open('t_s_utf8.txt','r',encoding='utf8')
tmp = f.readlines()
f.close()
# type and text 분리
def naive(tmp):    
    data=[]
    spam_all_text = []
    all_text=[]
    spam_cnt=0
    ham_cnt=0
    for k,i in enumerate(tmp):
        if(k>0):#첫줄에 이상한 글자때문에 자를려고
            aa = i.replace('\n','').lower()
            aa = aa.replace('.','').replace('?','').replace('!','')
            aa = aa.split('\t')
            
            data.append(aa) 
        
    data = np.array(data)
    
    train_types = data[:5000,0]
    train_text = data[:5000,-1]
    
    test_types = data[5000:,0]
    test_text = data[5000:,-1]   

    # 단어별로 쪼개기
    for i,j in zip(train_types,train_text):
        all_text += j.split(' ')
        if(i=='spam'):    
            spam_all_text += (j.split(' '))
            spam_cnt += 1
        elif(i == 'ham'):
            ham_cnt += 1
            
    q = Counter(spam_all_text).most_common(100) #  최빈값
    most_spam_word=[i[0] for i in q]
    most_spam_word_num=[i[1] for i in q]
    
    
    for i,j in enumerate(most_spam_word): #불용어 제거 ( stop word)
        for k in stopwords.words('english'):
            if j==k:
                
                del most_spam_word[i]
                del most_spam_word_num[i]

    plt.bar(most_spam_word,most_spam_word_num)
    plt.show()
    print(most_spam_word)
    spam_p = np.log(spam_cnt) - np.log(spam_cnt + ham_cnt) # spam 확률
    
    spam_word_c_p = [i/spam_cnt for i in most_spam_word_num] # 스팸 용어 조건부 확률
    s_word_c_p = 0
    for i in spam_word_c_p: #스팹 용어 조건부 확률 곱셈 즉, 스팸일때 그 단어들이 뜰 확률
        s_word_c_p += np.log(i)    
        
    spam_word_p = [i/len(all_text) for i in most_spam_word_num] # 스팸용어의 확률    
    s_word_p=0
    for i in spam_word_p:
        s_word_p += np.log(i) #스팸용어셋이 만들어질 확률
        
    # 언더 플로우 방지로 로그 값
    result = (s_word_c_p*spam_p)/s_word_p
    print('(',spam_p,' * ',s_word_c_p,')',' / ',s_word_p)
    print (np.exp(result))
    
# test start  
    
    test_all_text = []
    
    for i,j in zip(test_types,test_text):
        test_all_text += j.split(' ')
    # 문장마다 단어 쪼개서 불용어 없애기    
    tmpp=[1]*len(test_types)
    for i,j in enumerate(test_text):
        no = j.split(' ')
        for n,m in enumerate(no):            
            for k in stopwords.words('english'):
                if m==k:
                    del no[n]
        tmpp[i]=no
    
    # 공백 원소 없애기
    for i,j in enumerate(tmpp):
        for k,h in enumerate(j):
            if h == '':
                del tmpp[i][k]
            
#    test_p = [] # 테스트 각 문장에서 에러가 있을 확률
#    for i in tmpp:
#        test_tmp=[]
#        for j in i:
#            cnt=0
#            for k in most_spam_word:
#                if j==k:
#                    cnt += 1
#                
#                test_tmp.append(np.log(((cnt/(len(j)+10**20))+1e-9)))
#        test_p.append( sum(test_tmp))
    
    return np.exp(result),most_spam_word,most_spam_word_num,tmpp,test_types
    
res , m, m_n ,t,tp= naive(tmp)

for i,j in enumerate(tp):
    if j=='ham':
        tp[i]=1
    elif j=='spam':
        tp[i]=0

plt.close()

