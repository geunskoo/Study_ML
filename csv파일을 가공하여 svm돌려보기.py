#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import svm, metrics
import random, re


# In[2]:


#붓꽃의 데이터를 읽어들이기
csv = []


# In[8]:


with open('iris.csv','r',encoding = 'utf-8') as fp: #한줄 씩 읽어오기.
    for line in fp:
        line = line.strip() #줄 바꿈 제거
        cols = line.split(',') #쉼표로 컬럼을 구분
        
        #문자열 데이터를 숫자로 변환하기.
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$',n) else n
        cols = list(map(fn, cols))
        csv.append(cols)

#헤더 칼럼 제거하기.
del csv[0]

# 데이터를 섞어주기.
random.shuffle(csv)

#훈련(학습)데이터와 테스트 데이터로 분리하기.
#직접적으로 나눠보겠다.  사이킷런에서 제공하는 트레인 테스트 함수를 쓰면 위과정을
#수행할 수 있다.

total_len = len(csv)
train_len = int(total_len*2/3)

train_data = []
train_label = []

test_data = []
test_label = []

for i in range(total_len):
    data = csv[i][0:4]
    label = csv[i][4]
    if i<train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_data.append(label)

#학습 
clf =svm.SVC()
clf.fit(train_data,train_label)

#테스트
pre_label = clf.predict(test_data)

#정확도 구하기
ac_score = metrics.accuracy_score(test_label,pre_label)


# * CSV 파일을 읽어올때 Pandas를 이용하면 편리하다.

# 조금더 야무지게 파이썬 활용해서 데이터 분석하기 

# In[9]:


import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split


# In[10]:


pd.read_csv('iris.csv')


# In[30]:


#데이터와 레이블 분리하기
csv2=pd.read_csv('iris.csv')

csv2_data=csv2[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
csv2_label = csv2[['Name']]

#훈련데이터와 테스트 데이터로 분리하기.

Xtrain,Xtest,ytrain,ytest = train_test_split(csv2_data,csv2_label)

clf = svm.SVC()
clf.fit(Xtrain,ytrain)
y_predict = clf.predict(Xtest)

ac2_score = metrics.accuracy_score(ytest,y_predict)
print('정확도는 {}입니다'.format(ac2_score))


# In[ ]:




