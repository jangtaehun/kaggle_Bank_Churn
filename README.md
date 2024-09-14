### 👨‍🏫 Santander Customer Satisfaction - Machine Learning from Disaster
kaggle에서 제공하는 Binary Classification with a Bank Churn Dataset을 EDA와 model 학습을 통해 고객 이탈 확률을 예측하는 프로젝트

---
### ⏲️ 분석 기간
2024.09.10 - 2024.09.13

---

### 📝 소개
Binary Classification with a Bank Churn Dataset은 이전에 진행했던 Santander Customer Satisfaction data와 비슷하게 고객 이탈에 대한 예측을 하는 문제이다. 하지만 feature가 가려져 있는 것이 아니기 때문에 Feature Engineering을 진행할 때 feature들을 유용하게 이용할 수 있다.

따라서 이번엔 분석할 Binary Classification with a Bank Churn Dataset은 Santander Customer Satisfaction data와 titanic data를 섞어 놓은 데이터라고 할 수 있다.

---

### 프로젝트 개요
##### 📌 목표
kaggle에서 업로드한 overview는 다음과 같다.

![image](https://github.com/user-attachments/assets/c043d858-8be7-43d8-b19d-b0f2d9268575)

즉, 고객이 계속해서 서비스를 이용할지 아니면 계정을 닫고 서비스를 중단할지를 예측하는 것으로 이진분류 문제이다

##### 🖥️ 데이터셋 (Data Set)
이 프로젝트에서 사용한 데이터셋은 Kaggle에서 제공하는 다음 파일들로 구성되어 있습니다.
1. train.csv: 훈련 데이터셋, 특징들과 목표 변수를 포함.
2. test.csv: 테스트 데이터셋, 예측을 위해 사용될 데이터.
3. sample_submission.csv: 예측 결과를 제출하기 위한 샘플 파일.

---

##### 방법론
1. 문제에 대한 정보 수집
  * 문제 정의
2. Bank Churn Data set을 이용한 EDA 및 Data Cleaning
  * 공통 코드
  * 분석
    * Bank Churn Data set에 대한 기본적인 정보
    * feature 분석
    * Data cleaning
    * Feature Engineering
3. 모델 학습
  * XGBoost
  * LightGBM
  * CatBoost
  * Data Leakage
4. 결론
  * 한계점

---

### 문제에 대한 정보 수집
   #### 1. 문제 정의
Bank Churn Dataset은 kaggle에서 2024년을 맞이해 제공한 것으로 고객 이탈(Churn) 예측을 진행하는 것이다. 즉, 고객이 계속해서 서비스를 이용할지 아니면 계정을 닫고 서비스를 중단할지를 예측하는 것으로 이진분류 문제이다. 다만 이번 kaggle 대회는 이탈할 가능성을 나타내는 예측하는 것으로 제출할 때는 이진 분류의 확률 값을 제출해야 한다. 즉, 예측한 값이 0 또는 1과 같은 이진 값이 아닌, 0과 1 사이의 확률이어야 하므로 회귀처럼 보일 수 있지만, 근본적으로는 분류 문제이다.

제공되는 데이터는 train.csv, test.csv, sample_submission.scv 세 개의 파일로 train.csv를 토대로 test.csv의 고객 이탈 여부를 예측하는 문제이다. 이후 sample_submission.csv에 입력한 후 제출하는 해당 파일을 제출하는 것이다.

### Santander Customer Satisfaction data set을 이용한 EDA
   #### 1. 공통 코드
import libraries and files
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 110
pd.set_option('display.max_columns', None)

train_df = pd.read_csv("../../data/Bank_Churn_Dataset/train.csv")
test_df = pd.read_csv("../../data/Bank_Churn_Dataset/test.csv")
submission_df = pd.read_csv("../../data/Bank_Churn_Dataset/sample_submission.csv")
```
필자의 경우 파일을 따로 저장해 두는 폴더가 있기 때문에 그곳에 저장을 해두고 있다. 따라서 코드가 해당 폴더를 찾아가야 하기 때문에 위와 같이 작성된 것이다.

평가를 위한 함

![image](https://github.com/user-attachments/assets/6c67834d-b07c-4faa-adf9-85762112d42a)

kaggle에서 공지한 평가 방식은 위와 같다. ROC 곡선 아래의 면적(AUC)이 모델의 최종 성능 지표로 사용된다. 즉, AUC-ROC는 모델이 타겟 클래스(이탈 또는 유지)를 얼마나 잘 구분하는지 평가하는 중요한 지표이다. 따라서 오차행렬, 정확도, 정밀도, 재현율, F1 Score, AUC를 모두 확인할 것이다.

   #### 2. 분석
   ##### 1. Bank Churn Data set에 대한 기본적인 정보
```
train_df.shape
test_df.shape
submission_df.shape

(165034, 14)
(110023, 13)
(110023, 2)
```
위와 같이 출력된다. 즉, train_df로 모델을 학습을 한 후 test_df를 예측한 다음 submission_df에 입력해 제출하는 것이다.

train_df는 index와 id를 제외한 12개의 feature를 가지고 있으며 아래 dataframe과 같다.

![image](https://github.com/user-attachments/assets/7ba94325-baa0-41a4-bb4d-8eca5c37aeea)

각각 feature에 대한 설명은 다음과 같다.

* Surname: 성(이름)
* CreditScore: 신용 점수
* Geography: 거주 국가
* Gender: 성
* Age: 나이
* Tenure: 은행을 이용한 기간
* Balance: 계좌 잔액
* NumOfProducts: 이용하는 은행 상품의 수(ex. 예금,적금)
* HasCrCard: 신용카드 보유 여부
* IsActiveMember: 활성 회원 여부
* EstimatedSalary: 예상 연봉
* Exited: 이탈 여부

```
train_df.describe()
```
![image](https://github.com/user-attachments/assets/a21cdec6-6ad2-42ad-b97d-8ad6cc21f0ea)

대략적으로 볼 때 이상치가 보이지는 않다. 다만 CreditScore와 Age가 넓게 분포한 것은 확인할 수 있다. 따라서 범위를 특정해서 구분하는 것도 좋은 방법이라고 생각한다.

```
target_cnt = train_df['Exited'].count()
print(train_df['Exited'].value_counts())
train_df['Exited'].value_counts() / target_cnt

Exited
0    130113
1     34921

Exited
0    0.788401
1    0.211599
```
Exited에 대한  count를 보면 불균형적인 데이터인 것을 확인할 수 있다. 따라서 sampling 작업이 추가적으로 필요하다.

   ##### 2. Data cleaning
1. data set에 모든 값이 NaN인 컬럼을 확인 및 drop
```
all_nan_columns = train_df.columns[train_df.isna().all()].tolist()
print(f"모든 값이 NaN인 컬럼 개수: {len(all_nan_columns)}")

train_df.drop(columns=all_nan_columns, inplace=True, axis=1)
test_df.drop(columns=all_nan_columns, inplace=True, axis=1)
```
이미 앞에서 확인했듯 결과는 0개로 없다.

2. 고유값이 1인 컬럼 확인 및 drop
```
unique_one_columns = [col for col in train_df.columns if train_df[col].nunique() == 1]
print(f'고유값이 1인 컬럼 개수: {len(unique_one_columns)}')

train_df.drop(columns=unique_one_columns, inplace=True, axis=1)
test_df.drop(columns=unique_one_columns, inplace=True, axis=1)
```
3. 중복 데이터 drop
```
train_df.drop(['id', 'CustomerId'], axis=1, inplace=True)
test_df.drop(['id', 'CustomerId'], axis=1, inplace=True)
```
중복 데이터를 처리하기 전에 id와 CustomerId를 먼저 drop한 후 진행했다.
```
train_df.duplicated().sum()
```
54개의 중복된 데이터가 있다. 즉, 54개의 행이 다른 행과 동일한 값을 가지고 있다는 뜻이다. 중복된 데이터가 중요한 데이터인지 확인을 한 후에 drop을 해야 한다. 

예를 들어 설문 조사나 투표와 같은 데이터에서는 동일한 응답이 여러 번 있을 수 있으며, 이런 데이터는 중요한 값이기 때문이다. 반면 중복으로 수집된 데이터라면 삭제해도 되는 데이터이다. 따라서 확인이 필요하다.
```
duplicate_all = train_df[train_df.duplicated(keep=False)]
duplicate_all[duplicate_all['Surname']== 'Cunningham']
```
![image](https://github.com/user-attachments/assets/8dd98264-6b49-4a58-b600-b6a547c41653)

다음과 같이 확인할 수 있다. 대표적으로 Cunningham이란 성을 가진 사람의 데이터를 출력한 것으로 id와 CustomerId는 다르지만 다른 것은 모두 같다. 신용점수, 나이, 거주 국가, 성별, 나이, 은행 이용 기간, 계좌 잔액 등 전부 같다. 특히, 필자는 서로 다른 사람이 신용점수, 나이, 은행 이용 기간, 예상 연봉 모두 같다는 것은 불가능 하다 생각하기 때문에 삭제하기로 했다.
```
train_df = train_df.drop_duplicates()
```
4. 중복된 데이터 중에서 타겟 값(Exited)이 다른 데이터 찾기 - noise 찾기
```
y = train_df['Exited']
X = train_df.drop(['Exited'], axis=1)

train_with_target = pd.concat([X, y], axis=1)

duplicates = train_with_target.duplicated(keep=False)
duplicates_with_different_target = duplicates & (train_with_target.groupby(list(X.columns))['Exited'].transform('nunique') > 1)

noise = train_with_target[duplicates_with_different_target]
cleaned_train = train_with_target[~duplicates_with_different_target]

X = cleaned_train.drop('Exited', axis=1)
y = cleaned_train['Exited']
```
noise 데이터는 없다. 따라서 여기서 Data cleaning 작업은 마무리 하겠다.

   ##### 2. Data cleaning
1. 상관관계
숫자형 feature에 대해 상관관계를 보겠다. 아래 코드를 입력하면 아래 사진처럼 상관관계를 확인할 수 있다. 모든 컬럼이 자기 자신을 빼고 연관이 거의 없다.
```
numeric_df = train_df.select_dtypes(include=['float', 'int']).columns
numeric_df = train_df[numeric_df]

corr = numeric_df.corr()
corr.style.background_gradient(cmap='coolwarm')
```
![image](https://github.com/user-attachments/assets/2a424c73-87b9-490e-99e4-819990dbe076)

2. Age

나이는 은행 서비스를 계속 이용할지 안 할지를 결정하는 데 좋은 데이터라고 생각한다. 이유는 다음과 같다. 고객의 나이에 따라 금융 서비스에 대한 필요와 선호가 다르기 때문이다. 즉 젊은 층은 은행이나 금융 서비스를 적극적으로 비교하며, 더 나은 혜택이나 더 편리한 서비스를 제공하는 다른 금융 기관으로 이탈할 가능성이 크며, 나이가 많은 고객은 기존의 금융 서비스에 익숙해져 있으며, 서비스 변경에 따른 불편을 피하려고 하기 때문에 서비스를 유지하는 경향이 있을 수 있기 때문이다.

나이의 분포가 18~92까지 다양하게 있다. 따라서 연령대를 특정 구간으로 나누는 작업을 하겠다.
```
def get_age(age):
    cat = ''
    if age <= 23: cat = 'Student'
    elif age <= 39: cat = 'Young Adult'
    elif age <= 64: cat = 'Adult'
    else: cat = 'Elderly'        
    return cat

group_names = ['Student', 'Young Adult', 'Adult', 'Elderly']
 
train_df['Age_range'] = train_df['Age'].apply(lambda x : get_age(x))
test_df['Age_range'] = test_df['Age'].apply(lambda x : get_age(x))
```
```
age_range_counts = train_df.groupby('Age_range')['Exited'].value_counts(normalize=True).unstack()
age_range_counts
```
![image](https://github.com/user-attachments/assets/88fc4b86-3dd4-4007-89ce-1f41b5d70f16)

결과를 보면 필자가 예상한 것과 다르게 Adult에서 이탈 고객이 가장 많았고 다음으로 Elderly가 많았다. Student, Young Adult가 이탈 고객이 적었다. 따라서 다른 데이터를 확인해 볼 필요가 있다.

3. NumOfProduct

은행에서 이용하는 상품의 수이다. 상품의 수가 많으면 다른 은행에서도 거래를 하고 있을 가능성이 높아 이탈 가능성이 더 높을 수 있다. 물론 많은 상품을 이용하는 고객은 해당 은행과 더 깊은 관계를 형성하기 때문에 이탈할 가능성이 적을 수도 있다.
```
Products_exited_counts = numeric_df.groupby('NumOfProducts')['Exited'].value_counts(normalize=True).unstack()
Products_exited_counts
```
![image](https://github.com/user-attachments/assets/5342a783-c3c4-4c43-be1f-860eab26a794)

결과는 상품을 많이 사용할 수록 이탈하는 사람이 앞도적으로 높은 것을 확인할 수 있다. (본 데이터는 실제 은행 데이터가 아니기 때문에 실제와 다를 수 있다.) 특히, 1개를 이용할 때 역시 이탈율이 높다는 것도 확인할 수 있다.
```
train_df.groupby('Age_range')['NumOfProducts'].value_counts(normalize=True).unstack()
```
![image](https://github.com/user-attachments/assets/9a0dd1a3-09d9-4686-b7a4-325c2649029f)

연령대가 높을 수록 은행에서 이용하는 상품 수가 많다. 전체적으로 보면 전체 연령대에서 1~2개를 이용하는 비율이 높지만 이탈율이 가장 많은 Adult, Elderly에서 1개만 이용하는 비율이 높으며 3~4개를 이용하는 비율 역시 높다. 아래 표는 Adult, Elderly일 때 Student와 Young Adult 간의 이용 상품 수가 3~4개 일 때의 비율을 비교한 것이다. 이렇게 비교를 통해 Adult와 Elderly가 이탈 가능성이 더 높은 것을 확인할 수 있다.

<img width="716" alt="image" src="https://github.com/user-attachments/assets/bc8cc872-2ee9-464d-8a01-63fdbfa456b4">


4. Tenure - 은행 이용 기간

은행 이용 기간 역시 은행 서비스 이용에 있어 큰 영향을 줄 것으로 생각하고 있다. 그 이유는 신용 카드, 대출, 저축 계좌 등을 여러 해 동안 이용했을 수 있기 때문에 오랜 기간 이용할 경우 은행에 큰 실망을 하지 않는 이상 은행을 바꾸지 않기 때문이다.

```
tenure_exited_counts = numeric_df.groupby('Tenure')['Exited'].value_counts(normalize=True).unstack()
tenure_exited_counts
```
![image](https://github.com/user-attachments/assets/415f639a-b410-4f3d-8db4-15977144d4ba)

하지만 예상과 다르게 기간이 길 다고 이탈율이 높은 것은 아니였다. 물론 Tenure 즉, 이용 기간이 0일 때 이탈율이 가장 높지만 대체적으로 비슷하다.

이번에는 이용 기간 동안 얼마나 많은 상품을 이용했는 가를 계산해서 컬럼으로 만들어 보겠다.
```
train_df['Products_Per_Tenure'] =  train_df['Tenure'] / train_df['NumOfProducts']
test_df['Products_Per_Tenure'] =  test_df['Tenure'] / test_df['NumOfProducts']
```
```
products_per_tenure_counts = train_df.groupby('Products_Per_Tenure')['Exited'].value_counts(normalize=True).unstack()

products_per_tenure_counts.plot(kind='bar', stacked=True, figsize=(12, 6))

plt.xlabel('Products per Tenure')
plt.ylabel('Proportion of Exited')
plt.legend(title='Exited', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/ba9037b5-19ef-4848-861e-58fef06ece27)

모든 경우에서 적용되는 것은 아니지만 상품을 많이 이용했을 때 이탈율이 적고 적게 이용했을 때 이탈율이 높다는 것을 확인할 수 있다.

 

5. CreditScore

CreditScore 역시 Age와 비슷하게 다양하게 있기 때문에 특정 구간으로 나누겠다. 미국 신용 점수인 Fico Score를 기준으로 했다. 신용 점수가 높은 고객은 재정적으로 안정적이며, 더 많은 금융 상품(대출, 신용 카드 등)을 이용할 가능성이 높다. 또한, 은행과의 관계가 긍정적일 가능성이 크기 때문에 이탈 가능성이 낮을 수 있다.
```
def get_fico(age):
    cat = ''
    if age <= 579: cat = 'Poor'
    elif age <= 669: cat = 'Not_Good'
    elif age <= 799: cat = 'Very_Good'
    else: cat = 'Excellent'        
    return cat

group_names = ['Poor', 'Not_Good', 'Very_Good', 'Excellent']
 
train_df['Fico_Score'] = train_df['CreditScore'].apply(lambda x : get_fico(x))
test_df['Fico_Score'] = test_df['CreditScore'].apply(lambda x : get_fico(x))
```
```
credit_exited_counts = train_df.groupby('Fico_Score')['Exited'].value_counts(normalize=True).unstack()
credit_exited_counts
```
![image](https://github.com/user-attachments/assets/50a9b73f-4f35-4791-ae6c-e2f05eca6077)

예상과 다르게 특별하게 큰 차이가 보이지는 않지만 Poor 구간에서 가장 높은 이탈율을 보여주고 있다.

6. Balance(계좌 잔액), EstimatedSalary(예상 연봉)

계좌 잔액과 예상 연봉은 같이 비교를 하겠다. 이유는 대부분 연봉이 높을 수록 계좌 잔액이 많기 때문이다. 두 데이터는 고객의 재정적 안정성을 나타내는 지표로 중요하다.
```
train_df['Balance_to_EstimatedSalary'] = train_df['Balance'] / train_df['EstimatedSalary']
test_df['Balance_to_EstimatedSalary'] = test_df['Balance'] / test_df['EstimatedSalary']
```
```
credit_exited_counts = train_df.groupby('Balance_to_EstimatedSalary')['Exited'].value_counts(normalize=True).unstack()
credit_exited_counts
```
![image](https://github.com/user-attachments/assets/51120ece-aae2-4b1f-a7d6-afc6848feb4c)

```
nan_exited_1 = credit_exited_counts[credit_exited_counts[1].isna()]
nan_exited_1
```
![image](https://github.com/user-attachments/assets/7107020c-3ff9-4c69-9f07-27e26055ff92)

많은 행에서 1이 NaN 값이다. 하지만  보유 금액과 예상 연봉이 이탈율과는 크게 연관이 없는 것 같다.

7. category feature
```
sum_columns = ['Surname', 'Geography', 'Gender']
```
위의 세 개의 feature가 아직 사용하지 않고 남은 feature다 Surname의 경우 성(이름)으로 국가, 성별과 결합하면 새로운 데이터를 줄 수 있다고 예상하고 있어서 결합하기로 했다.
```
train_df['Surename_Geography_Gender'] = train_df[sum_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
test_df['Surename_Geography_Gender'] = test_df[sum_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
train_df
```
결과적으로 아래와 같이 feature가 만들어 졌다.
![image](https://github.com/user-attachments/assets/91b92fa3-bb9a-455e-a22e-c082c48e65e2)




