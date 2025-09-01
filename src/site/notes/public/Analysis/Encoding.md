---
{"dg-publish":true,"permalink":"/public/analysis/encoding/","tags":["encoding","prerocessing","nominal_data","LabelEncoding","categorical_data","OneHotEncoding"],"created":"2025-08-20T13:39:15.632+09:00","updated":"2025-08-20T14:38:17.370+09:00"}
---


# Nominal data

- LabelEncoding
	- 동작방식
		- `LabelEncoder`는 **문자열(카테고리 값)을 숫자로 단순히 치환**합니다.
		- 이 숫자는 **사전순(알파벳 오름차순 또는 유니코드 순서)** 으로 부여됩니다.
		- 순서를 부여하지만, 그 숫자가 '우열'이나 '크기 비교'를 의미하지는 않습니다.
		- 단순히 "이 값은 몇 번째 클래스인가"를 나타내는 인덱스일 뿐입니다.
	- 주의
		- 트리 기반 모델(Decision Tree, RandomForest, XGBoost 등)에서는 큰 문제가 되지 않습니다. 이 값이 단순 분할 기준으로만 쓰이기 때문입니다.
	    - 그러나 선형 모델(Logistic Regression, Linear Regression 등)에서는 **숫자 간의 크기 차이가 의미 있는 값처럼 잘못 해석**될 수 있습니다.
		- 예: `red=2`, `blue=0`, `green=1` → `red`가 `blue`보다 "2만큼 더 크다"는 식으로 오해됨.
		- Ordinal 데이터인 경우 `LabelEncoder` 대신 **OrdinalEncoder**를 써서 직접 순서를 지정하는 것이 적절합니다.

# Ordinal data

- OrdinalEncoder
	- 동작방식
		- 이 클래스는 각 범주의 순서를 지정할 수 있어 순서형 데이터에 적합합니다.
		- 

```python
eocoder_train = OrdianlEncoder(categories = [['Preschool', '1st-4th', 'Doctorate']]) # 인코더 초기화 및 범주 순서 설정
train_ex3['education'] = encoder_train.fit_transform(train_ex3[['education']]).astype(int) # fit_transofrm 메서드를 사용하여 교육 수준 데이터에 OrdinalEncoder 를 적용하고, 결과를 정수형으로 반환합니다.

display(encoder_train.categories_) # 인코딩 순서를 확인
display(train_ex3.head())
display(train_ex3['education'].dtype()) # 데이터 타입 체크
display(train_ex3['education'].unique()) # unique 값 유지 여부 체크



```

# 딕셔너리를 이용한 하드 인코딩 Direct Mapping

- map()
```python

order_map = {'낮음' : 1, '중간' : 2, '높음' : 3}
data['변수명'] = data['변수명'].map(order_map)

```

- replace()
```python
education_map = {'Preschool': 0, '1st-4th' : 1, 'Doctorate', 15}
train_ex2['education'] = train_ex2['education'].replace(education_map)

```

- 수정 후 
```python
train_ex2.head() # 잘 바뀌었는지
train_ex2['education'].dtype() # 희망하는 data type 인지
train_ex2['education'].unique() # unique 동일한지 

```

# 학습 데이터에 없는 범주 올바르게 처리하기

```python

test_ex2 = test_used_car.copy()

# train_used_car 데이터셋의 라벨 인코더 클래스에 '기타' 범주를 다시 추가
for le in label_encoders.values():
    le.classes_ = np.append(le.classes_, 'Other')

# 새로운 범주값을 '기타'로 매핑하고 라벨 인코더를 적용하는 과정을 하나의 루프로 합치기
for col in categorical_var:
    # 새로운 값들을 '기타'로 매핑
    test_ex2[col] = test_ex2[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'Other')
    # 라벨 인코더 적용
    test_ex2[col] = label_encoders[col].transform(test_ex2[col])

# 변환된 test_used_car 데이터셋 보여주기
display(test_ex2.head())

```


# Nominal Data

- OneHotEncoding
	- 학습 데이터 없는 범주 처리 방법
	- 더미변수 drop 옵션
		- 컬럼 간에 상호 의존적 관계가 형성되는 '다중공선성'을 배제하기 위해 사용되며, 다중공선성이 발생하는 경우 모델의 예측력 저하, 해석을 어렵게 만드는 원인이 됩니다.
		- drop_first = True



	- sparse_output 파라미터
		- 개념
			- `OneHotEncoder`는 원래 **희소 행렬(sparse matrix)** 형태로 결과를 반환했습니다.
			- 이유: One-hot은 대부분 값이 0이기 때문에, 모든 원소를 저장하는 dense(밀집) 행렬은 메모리 낭비가 심하기 때문입니다.
			- 예: 10000개의 샘플 × 100개의 카테고리 → 실제로는 대부분 0.

		-  `sparse` vs `sparse_output`
			- 예전 버전 (`scikit-learn < 1.2`)에서는 `sparse=True/False` 옵션을 사용했습니다.  
			- 최신 버전 (`scikit-learn >= 1.2`)에서는 `sparse` 대신 `sparse_output`을 사용하도록 바뀌었고, `sparse`는 **deprecated(사용 중단 예정)** 상태입니다.

		- 동작방식
			- `sparse_output=True` (기본값)  
			    → **희소 행렬(scipy.sparse CSR matrix)** 반환
			- `sparse_output=False`  
			    → **numpy 배열(ndarray, dense matrix)** 반환

		- 언제 어떻게 쓰냐
			- **데이터 차원이 크다 (범주가 많다)** → `sparse_output=True` (메모리 효율 ↑)
		    - **작은 데이터이고 numpy array가 필요하다** (예: Pandas DataFrame과 병합할 때, 시각화 등) → `sparse_output=False`
	```python
	from sklearn.preprocessing import OneHotEncoder
	
	data = [["red"], ["green"], ["blue"], ["green"]]
	
	# 1) 기본값 (sparse_output=True)
	enc1 = OneHotEncoder(sparse_output=True)
	X1 = enc1.fit_transform(data)
	print(type(X1))   # <class 'scipy.sparse._csr.csr_matrix'>
	
	# 2) dense matrix로 출력
	enc2 = OneHotEncoder(sparse_output=False)
	X2 = enc2.fit_transform(data)
	print(type(X2))   # <class 'numpy.ndarray'>
	print(X2)
	```

	- handle_unknown 파라미터
		- handle_unknown = 'ignore'
			- 훈련 데이터에 없는 범주가 테스트 데이터에 나타날 때 이를 무시하고 애러를 발생시키지 않도록 설정
			- 새로운 범주 데이터에 대해서 오류 없이 수행할 수 있으나, 신규 발견된 범주의 중요도 등이 무시될 수 있음


- Binary Encoding

이렇게 고유 범주의 갯수가 많은 명목형 변수를 인코딩할 수 있는 대표적인 기법이 바이너리 인코딩(binary encoding)입니다.  
이 기법은 데이터의 차원을 효과적으로 줄이면서 각 범주를 명확하게 표현할 수 있는 장점을 가지고 있습니다.

바이너리 인코딩은 각 범주를 레이블 인코딩으로 변환한 후, 이진수로 표현하여 범주의 수에 비례하지 않는 고정된 길이의 열을 생성합니다. 이는 차원의 증가를 상당히 제한하며, 동시에 각 범주를 효율적으로 표현할 수 있게 합니다.

- 범주형 데이터를 숫자로 변환할 때, 단순히 “인덱스 번호”로 바꾸지 않고,  
    그 인덱스를 **이진수(binary number)** 로 변환해서 컬럼을 생성하는 방식입니다.
- 일종의 **LabelEncoding + 이진수 변환 + 분리**라고 보시면 됩니다.


```python
import category_encoders as ce

train_ex5 = train_income.copy()

train_ex5['occupation'].fillna('Unknown', inplace=True)

# Creating a BinaryEncoder instance
encoder_ex5 = ce.BinaryEncoder(cols=['occupation'])

# Fitting and transforming the 'occupation' column
train_encoded_ex5 = encoder_ex5.fit_transform(train_ex5['occupation'])

encoded_columns_ex5 = train_encoded_ex5.columns

# Creating a DataFrame from the binary encoded data
train_encoded_df5 = pd.DataFrame(train_encoded_ex5, columns=encoded_columns_ex5)
train_ex5 = pd.concat([train_ex5, train_encoded_df5], axis=1).drop(['occupation'], axis=1)

display(encoded_columns_ex5)
display(train_ex5.head())

```

카테고리: `["red", "green", "blue", "yellow"]`

1. 먼저 Label Encoding:
    - red=1, green=2, blue=3, yellow=4
        
2. 이 값을 이진수로 변환:
    - 1 → 01
    - 2 → 10
    - 3 → 11
    - 4 → 100
        
3. 자리수를 컬럼으로 나눔:
```
red    → [0, 0, 1]
green  → [0, 1, 0]
blue   → [0, 1, 1]
yellow → [1, 0, 0]

```

특징
- **장점**: OneHot보다 차원이 훨씬 줄어듦 → 범주가 많을 때 유리
- **단점**: 변환된 숫자가 원래 카테고리 간 의미 없는 “유사성”처럼 보일 수 있음 (완전 독립 표현이 아님)
- **적합한 경우**: 범주 개수가 매우 많을 때 (예: 수천 개 이상의 값)



| 구분         | **LabelEncoding**                  | **OneHotEncoding**                         | **OrdinalEncoding**         |
| ---------- | ---------------------------------- | ------------------------------------------ | --------------------------- |
| **방식**     | 카테고리를 숫자 인덱스로 변환                   | 각 카테고리를 0/1 이진 벡터로 변환                      | 카테고리를 **정해진 순서대로** 숫자로 변환   |
| **예시 데이터** | `["red", "blue", "green"]`         | `["red", "blue", "green"]`                 | `["low", "medium", "high"]` |
| **결과**     | `blue=0, green=1, red=2`           | `blue=[1,0,0], green=[0,1,0], red=[0,0,1]` | `low=0, medium=1, high=2`   |
| **순서 의미**  | 없음 (사전순 인덱스)                       | 없음 (모두 동등)                                 | 있음 (사용자가 지정한 순서대로)          |
| **장점**     | 간단, 숫자 하나로 표현 가능                   | 범주 간 크기 의미 왜곡 없음                           | 순서를 가진 데이터 표현에 적합           |
| **단점**     | 숫자의 크기 차이가 의미 있는 것처럼 오해될 수 있음      | 차원이 많아질 수 있음 (고차원 문제)                      | 순서를 잘못 지정하면 오해 발생           |
| **적합한 경우** | 트리 기반 모델 (RandomForest, XGBoost 등) | 대부분의 모델 (선형/비선형)                           | 순서형 데이터 (학력, 만족도 등)         |


# 어떻게 쓸까

- **순서 없는 범주형 변수** → `OneHotEncoding` 권장
- **순서 있는 범주형 변수** → `OrdinalEncoding`
- **단순히 인덱스 부여만 필요하거나 트리 모델용** → `LabelEncoding`


# 주의할 점

순서형 범주형 변수, 특히 많은 고유 카테고리를 가진 경우, 데이터 분석과 모델링에서 중요하고 세심한 처리가 필요한 부분입니다. 
이러한 변수들은 고유한 범주(카테고리)가 많기 때문에, 이들을 어떻게 효과적으로 인코딩하는지는 모델의 성능에 결정적인 영향을 미칠 수 있습니다. 따라서, 순서형 범주형 변수를 다룰 때는 이에 적합한 인코딩 방법을 선택하는 것이 필수적입니다.

[1] 고유 범주 수 축소:
순서형 범주형 변수에 대한 첫 번째 단계는 가능하다면 범주의 수를 줄이는 것입니다.
이는 통계적 분석이나 도메인 지식을 활용하여 유사한 범주를 통합함으로써 이루어질 수 있습니다.
예를 들어, 빈도수가 낮은 범주들을 더 큰 범주로 통합하거나, 의미론적으로 유사한 범주들을 하나로 합치는 것입니다.
이는 모델이 데이터를 더 잘 이해하고, 복잡도를 감소시키는 것에 매우 '실질적'인 영향을 줄 수 있습니다.

[2] 의미 있는 수동 인코딩:
순서형 범주형 변수의 경우, 단순한 Label Encoding보다는 변수의 순서나 등급을 반영할 수 있는 Direct Mapping이나 Ordinal Encoder를 사용하는 것이 좋습니다. 
특히 Linear Regression, Ridge, Lasso와 같은 선형 모델에서는 이러한 순서 정보를 잘 반영하지 않으면 변수 간의 관계를 제대로 파악하지 못해 모델의 성능이 왜곡될 수 있으므로 순서를 반영한 인코딩이 필요합니다.

이러한 인사이트를 바탕으로, 순서형 범주형 변수를 다룰 때는 해당 변수의 성격을 고려하여 적절한 인코딩 전략을 선택하는 것이 중요합니다. 
이를 통해 모델의 성능을 최적화하고, 데이터 분석의 정확도를 높일 수 있습니다. 범주형 변수의 인코딩은 데이터 과학의 핵심적인 부분이므로, 이에 대한 이해와 적절한 적용은 데이터 분석의 성공에 큰 기여를 할 것입니다.


범주형 변수의 인코딩은 데이터 전처리에서 중요한 단계입니다, 특히 많은 범주를 가진 명목형 변수의 경우, 매우 신중하게 고유 카테고리의 영향도 및 분석 과정이 매우 중요합니다. 이러한 문제를 해결하는 데 있어 다음과 같은 전략을 제안합니다:

- **범주의 간소화:** 데이터 탐색(EDA)과 도메인 전문 지식을 활용하여 유사한 범주를 통합하고, 빈도가 낮은 범주를 병합하는 접근법은 범주의 수를 효과적으로 줄일 수 있습니다. 이러한 방법은 단순히 모델의 복잡성을 감소시키는 것을 넘어, 인코딩 기법 자체가 미치는 영향보다 더 큰 향상을 모델 성능에 가져올 수 있습니다.
    
- **바이너리 인코딩 적용:** 간소화된 범주 구조를 가진 후에는 바이너리 인코딩 방식을 적용하는 것이 권장됩니다. 바이너리 인코딩은 다수의 범주를 가진 변수에 대해 원-핫 인코딩에 비해 더 메모리 효율적이며, 이진 수 조합을 통해 각 범주를 표현함으로써 생성되는 새로운 변수의 수를 최소화합니다.
    
- **고급 인코딩 기법 고려:** 본 학습에서는 다루지 않았지만, 타깃 인코딩(Target Encoding)과 같은 고급 인코딩 기법도 주목해야 합니다. 이러한 기법은 종속 변수(타깃)와의 관계를 기반으로 각 범주를 인코딩하며, 데이콘 대회 혹은 kaggle 대회에서 인기 있는 기법으로 자주 사용되고 있습니다.


- Target Encoding

- 개념
	- 카테고리 자체를 **목표 변수(target)** 와의 통계적 관계로 변환하는 방식입니다.
	- 보통은 **카테고리별 평균 타깃값**을 숫자로 치환합니다.

- 예시
	- 문제: 고객의 `지역(region)`에 따라 `구매여부(y=0/1)` 예측

|region|y|
|---|---|
|A|1|
|A|0|
|B|1|
|B|1|
|C|0|

1. 각 지역별 `y` 평균 계산
    
    - A → (1+0)/2 = 0.5
        
    - B → (1+1)/2 = 1.0
        
    - C → (0)/1 = 0.0
        
2. region을 평균값으로 치환
    
    `A → 0.5 B → 1.0 C → 0.0`
    

- 특징
	- **장점**: 고차원 범주형 데이터에도 사용 가능, 모델에 직접 target 관련 정보 반영 → 예측력 ↑ 
	- **단점**: target과 직접 연결되므로 **데이터 누수(target leakage)** 위험
	    - 예: 학습 데이터 전체 평균을 사용하면 모델이 미래 정보를 미리 알아버림
	        
	- **적합한 경우**: 범주 개수가 많고, 범주별로 데이터 수가 충분한 경우
	- **보완 방법**: K-fold 평균, smoothing, leave-one-out 방식 사용


	- **Binary Encoding**은 **차원 축소용**
	- **Target Encoding**은 **성능 강화용** (하지만 누수 방지 조심)
