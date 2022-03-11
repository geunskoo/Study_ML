# 목표
* 지도학습 알고리즘 중 회귀에 대해서 알아보자
* 주어진 농어의 특성들을 이용하여 새로운 특성(무게)를 예측해보자!

## ML에 필요한 패키지 삽입


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
```

## 데이터 가공


```python
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
```


```python
plt.scatter(perch_length,perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![output_5_0](https://user-images.githubusercontent.com/97498405/157837755-b63e4319-0660-418e-9103-c9498cebe2f1.png)
    



```python
train_data,test_data,train_target,test_target = train_test_split(perch_length,perch_weight,random_state=42)
```


```python
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
train_target = train_target.reshape(-1,1)
test_target = test_target.reshape(-1,1)
```

## 결정계수($R^2$)


```python
#사이킷런 k - 최근접 이웃 알고리즘 회귀
from sklearn.neighbors import KNeighborsRegressor
```


```python
knr = KNeighborsRegressor()
knr.fit(train_data,train_target)
knr.score(test_data,test_target)
```




    0.992809406101064



* 분류 모델의 .score()멤버 함수 -> test데이터를 중 몇개 분류를 잘해서 맞추었는가?
* 회귀 모델의 .score()멤버 함수 -> 결정계수 $R^2$로 평가한다.
> $R^2$ = 1 - $(타킷 - 예측)^$ 2의 합 / $(타깃-평균)^2$의 합

* sklearn.metrics 패키지에서 mean_absolute_error
타깃과 예측값의 `절대값 오차` 의 평균


```python
from sklearn.metrics import mean_absolute_error

# 예측값을 만든다.
test_predict = knr.predict(test_data)

# 타깃과 예측값의 절대값오차의 평균!
mean_absolute_error(test_target,test_predict)
```




    19.157142857142862



평균 19g정도 오차가 존재하는 것을 알 수 있다!

## 과대적합 & 과소적합

### 과소적합 ?


```python
print(knr.score(train_data,train_target))
print(knr.score(test_data,test_target))
```

    0.9698823289099254
    0.992809406101064


**과대 적합** : 훈련 세트 score **>** 테스트 세트 score (모델이 훈련 데이터에 너무 잘 맞지만 일반성이 떨어진다)
                                    
**과소 적합** : 훈련 세트 score **<** 테스트 세트 score (모델이 너무 단순하여 제대로 된 값 측정을 못하고 있다!)

### 모델을 조금 더 복잡하게 만들어 보자!

k 최근접 이웃 회귀에서 모델의 복잡도를 올리는 간단한 방법

**k의 갯수를 줄이는 것 !!**
훈련세트의 국지적인 패턴에 민감해 질 것이다!


```python
knr.n_neighbors = 3 #defalut = 5

#모델 재훈련
knr.fit(train_data,train_target)

print(knr.score(train_data,train_target))
print(knr.score(test_data,test_target))
```

    0.9804899950518966
    0.9746459963987609


훈련 세트의 결정계수는 다소 증가하였고, 테스트 세트의 결정계수는 다소 감소하였다 !

## 1.5 평가!

```python
# 농어 50cm / 1.5kg 을 제대로 예측할 수 있을까 ?

fish_weight = knr.predict([[50]])
print(int(fish_weight)/1000,'Kg 입니다!')
```

```
1.033 Kg 입니다!
```

1.5 kg의 아주 실한 농어 인데, 무게의 1/3이 사라졌다 ?!



### 1.5.1 오류 찾기

```python
# 예측값에 사용된 이웃 데이터 정보구하기
distances, indexes = knr.kneighbors([[50]])


# 훈련데이터
plt.scatter(train_data,train_target)
# 사용된 이웃 데이터 3개
plt.scatter(train_data[indexes],train_target[indexes],marker='D')

# 예측된 값.
plt.scatter(50,1033,marker='^')
plt.show()
```

![output_27_0](https://user-images.githubusercontent.com/97498405/157841206-07d7cfe8-0705-4d40-8d17-dd73b0a6faf7.png)

이웃한 데이터를 참고 하기 때문에 이웃한 데이터보다 큰 길이를 가진 농어라 하더라도

무게는 1033g에서 늘지 않는다..!

..다른 방법은 없을까 ?!
