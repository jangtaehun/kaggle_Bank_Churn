### 👨‍🏫 Santander Customer Satisfaction
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

### Bank Churn Data set을 이용한 EDA 및 Data Cleaning
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

연령대가 높을 수록 은행에서 이용하는 상품 수가 많다. 전체적으로 보면 전체 연령대에서 1-2개를 이용하는 비율이 높지만 이탈율이 가장 많은 Adult, Elderly에서 1개만 이용하는 비율이 높으며 3-4개를 이용하는 비율 역시 높다. 아래 표는 Adult, Elderly일 때 Student와 Young Adult 간의 이용 상품 수가 3~4개 일 때의 비율을 비교한 것이다. 이렇게 비교를 통해 Adult와 Elderly가 이탈 가능성이 더 높은 것을 확인할 수 있다.

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

### 모델 학습
XGBoost와 LightGBM은 CatBoost와 다르게 라벨링 작업을 해야한다. 따라서 XGBoost와 LightGBM에 적용할 공통 코드를 먼저 작성하겠다.
```
y = train_df['Exited']
X = train_df.drop(['Exited','Surname'], axis=1)
test_df = test_df.drop(['Surname'], axis=1)
```
먼저 Surname을 drop하겠다. Surname은 train 데이터에 없는 것이 test 데이터에는 포함되어 있을 수 있기 때문에 Lable Encoding을 할 때 LabelEncoder.classes_ 을 이용하는 방법도 있지만 처음 보는 데이터는 잘못 학습할 수 있기 때문에 제거하기로 했다. 
```
X.drop(['Surename_Geography_Gender'], axis=1, inplace=True)
test_df.drop(['Surename_Geography_Gender'], axis=1, inplace=True)
```
위 코드 역시  Surname이 포함된 feature를 제거하는 것으로 이유는 위와 같다.
```
numeric_X = X.select_dtypes(include=['float', 'int']).columns
category_X = X.select_dtypes(include=['object']).columns
```
스케일링과 인코딩을 위해 numeric feature와 아닌 feature를 구분했다. 이후 밑에 있는 코드와 같이 스케일링과 인코딩을 진행했다.
```
# scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[numeric_X] = scaler.fit_transform(X[numeric_X])
test_df[numeric_X] = scaler.transform(test_df[numeric_X])

X = pd.DataFrame(X, columns=columns)
test_df = pd.DataFrame(test_df, columns=columns)

# label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in category_X:
    encoder = LabelEncoder()
    encoder.fit(X[i])
    
    X[i] = encoder.transform(X[i])
    test_df[i] = encoder.transform(test_df[i])
```
   #### 1. XGBoost
```
num_folds=5
n_est=3500
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
```
```
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, plot_importance

xgb_search_space = {'max_depth': hp.quniform('max_depth', 2, 15, 1), 
                    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.95),
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)}

def objective_func(search_space):
    xgb_clf = XGBClassifier(n_estimators=100,
                            max_depth=int(search_space['max_depth']),
                            min_child_weight=int(search_space['min_child_weight']),
                            colsample_bytree=search_space['colsample_bytree'],
                            learning_rate=search_space['learning_rate'],
                            early_stopping_rounds=30,
                            eval_metric='logloss',
                           random_state=RANDOM_STATE)
    
    roc_auc_list= []
    kf = KFold(n_splits=5)
    
    for tr_index, val_index in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[tr_index], y_train.iloc[tr_index]
        X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]
        
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        score = roc_auc_score(y_val, xgb_clf.predict_proba(X_val)[:, 1])
        roc_auc_list.append(score)
    return -1 * np.mean(roc_auc_list)

trials = Trials()
best = fmin(fn=objective_func,
            space=xgb_search_space,
            algo=tpe.suggest,
            max_evals=50, # 최대 반복 횟수를 지정합니다.
            trials=trials,
            rstate=np.random.default_rng()
           )
print('best:', best)

xgb_clf = XGBClassifier(n_estimators=500, learning_rate=round(best['learning_rate'], 5),
                        max_depth=int(best['max_depth']), min_child_weight=int(best['min_child_weight']), eval_metric="logloss",
                        colsample_bytree=round(best['colsample_bytree'], 5), random_state=RANDOM_STATE, verbose=False)

xgb_clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)])
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1])
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
```
```
pred = xgb_clf.predict(X_train) 
proba = xgb_clf.predict_proba(X_train)[:, 1]

best_rf_pred = xgb_clf.predict(X_test) 
best_rf_proba = xgb_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_train, pred, proba)
get_clf_eval(y_test , best_rf_pred, best_rf_proba)
```
optuna와 K-Fold를 이용해 베스트 파라미터를 구한 다음 결과 값을 확인해 보면 다음과 같다.
```
[[87041  4002]
 [10049 14394]]
정확도: 0.8783, 정밀도: 0.7825, 재현율: 0.5889,    F1: 0.6720, AUC:0.9102
오차 행렬
[[36944  2074]
 [ 4716  5760]]
정확도: 0.8628, 정밀도: 0.7353, 재현율: 0.5498,    F1: 0.6292, AUC:0.8869
```
평가 지표인 AUC는 train 데이터에 대한 AUC 값이 너무 높다면(예: 0.99 이상), 모델이 과적합될 가능성이 크기 때문에 test 데이터에 대한 AUC 값이 중요다. 즉, test 데이터에서 값이 높을수록, 모델이 일반화된 성능을 가지고 있다고 볼 수 있다. 결과를 보면 평가 기준인 AUC가 좋게 나왔으며 train과 test 세트에서 큰 차이가 나지 않어 과적합은 아니다. 따라서 좋은 모델이라고 볼 수 있다. 하지만 재현율이 둘 다 낮은 편이므로, 실제 이탈(Exited = 1)을 놓치는 경우가 많을 수 있다.
```
test_preds = np.empty((num_folds, len(test_df)))
auc_vals = []

folds = StratifiedKFold(n_splits=num_folds, random_state=RANDOM_STATE, shuffle=True)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    y_pred_val = xgb_clf.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print(f"AUC for fold {n_fold}: {auc_val}")
    auc_vals.append(auc_val)
    
    y_pred_test = xgb_clf.predict_proba(test_df)[:, 1]
    test_preds[n_fold, :] = y_pred_test
    print("----------------")

y_pred = test_preds.mean(axis=0)

print(f"최종 예측값 (y_pred): {y_pred}")
```
다음으로 k-fold를 이용해 데이터를 5개로 나눈 다음 위에서 학습한 베스트 파라미터를 통해 각 케이스마다 test 데이터를 예측한 후 test_preds에 저장했다. 예측한 확률들을 평균을 낸 다음 아래와 같이 제출 파일의 Exited에 입력한 후 제출했다.
```
submission_df['Exited'] = y_pred
submission_df.head()
submission_df.to_csv("submission.csv",index=False)
```
kaggle에 제출하면 다음과 같이 결과가 나온다.
![image](https://github.com/user-attachments/assets/a7c4b50f-6dbb-4978-9b71-de9f7545cdc3)

   #### 2. LightGBM
```
num_folds=5
n_est=3500
```
```
import optuna

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }

    lgb_model = LGBMClassifier(**param, random_state=RANDOM_STATE, verbose=-1)
    lgb_model.fit(X_train, y_train, feature_name=['f' + str(i) for i in range(X_train.shape[1])])
    y_val_pred = lgb_model.predict(X_test)
    f1 = f1_score(y_test, y_val_pred, pos_label=1) 
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

best_params = study.best_params
print("Best params: ", best_params)

best_lgb_model = LGBMClassifier(**best_params, random_state=RANDOM_STATE)
best_lgb_model.fit(X_train, y_train, feature_name=['f' + str(i) for i in range(X_train.shape[1])])
```
```
pred = best_lgb_model.predict(X_train) 
proba = best_lgb_model.predict_proba(X_train)[:, 1]

best_rf_pred = best_lgb_model.predict(X_test) 
best_rf_proba = best_lgb_model.predict_proba(X_test)[:, 1]

get_clf_eval(y_train, pred, proba)
get_clf_eval(y_test , best_rf_pred, best_rf_proba)
```
XGBoost와 같이 optuna와 K-Fold를 이용해 베스트 파라미터를 구한 다음 결과 값을 확인해 보면 다음과 같다.
```
오차 행렬
[[93819 10230]
 [ 5401 22534]]
정확도: 0.8816, 정밀도: 0.6878, 재현율: 0.8067,    F1: 0.7425, AUC:0.9352
오차 행렬
[[34992  4026]
 [ 2258  8218]]
정확도: 0.8730, 정밀도: 0.6712, 재현율: 0.7845,    F1: 0.7234, AUC:0.9259
```
과적합되지 않고 좋은 일반화 성능을 보이고 있다. XGBoost보다 재현율이 높아졌지만 정밀도가 낮아졌다.
```
test_preds = np.empty((num_folds, len(test_df)))
auc_vals = []

folds = StratifiedKFold(n_splits=num_folds, random_state=RANDOM_STATE, shuffle=True)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # 최적화된 파라미터로 LightGBM 모델 학습
    best_lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # 검증 데이터에 대한 예측
    y_pred_val = best_lgb_model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print(f"AUC for fold {n_fold}: {auc_val}")
    auc_vals.append(auc_val)
    
    # 테스트 데이터에 대한 예측을 저장
    y_pred_test = best_lgb_model.predict_proba(test_df)[:, 1]
    test_preds[n_fold, :] = y_pred_test
    print("----------------")

# 모든 fold에서의 테스트 예측값 평균 계산
y_pred = test_preds.mean(axis=0)

print(f"최종 예측값 (y_pred): {y_pred}")
```
```
submission_df['Exited'] = y_pred
submission_df.head()
submission_df.to_csv("submission.csv",index=False)
```
![image](https://github.com/user-attachments/assets/adad2008-68ca-4bc3-96a5-4f88cc0a1668)

결과는 XGBoost보다 살짝 높은 점수를 얻을 수 있다.

   #### 3. CatBoost
```
y = train_df['Exited']
X = train_df.drop(['Exited'], axis=1)
```
CatBoost는 XGBoost와 LightGBM과 같이 인코딩을 따로 안 해줘도 된다. 따라서 위와 같이 target만 X, y로 분리했다.
```
numeric_X = X.select_dtypes(include=['float', 'int']).columns
```
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[numeric_X] = scaler.fit_transform(X[numeric_X])
test_df[numeric_X] = scaler.transform(test_df[numeric_X])

X = pd.DataFrame(X, columns=columns)
test_df = pd.DataFrame(test_df, columns=columns)
```
스케일링을 위해 numeric feature만 따로 저장한 후 스케일링을 진행했다.
```
cat_features = np.where(X.dtypes != np.float64)[0]
```
CatBoost에서 float형이 아닌 컬럼을 알려줘야 인코딩을 따로 진행하지 않아도 처리가 가능하다. 따라서 따로 컬럼의 index를 추출했다. CatBoost가 좋은 점은 test 데이터에만 존재하는 데이터 즉, 새로운 카테고리 값을 처리할 수 있는 고유한 방식이 있다는 것이다.

```
num_folds=5
n_est=3500
```
```
folds = StratifiedKFold(n_splits=num_folds, random_state=RANDOM_STATE, shuffle=True)
test_preds = np.empty((num_folds, len(test_df)))
auc_vals=[]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[valid_idx], y.iloc[valid_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    clf = CatBoostClassifier(eval_metric='AUC', learning_rate=0.03, iterations=n_est)
    clf.fit(train_pool, eval_set=val_pool, verbose=300)
    
    y_pred_val = clf.predict_proba(X_val[columns])[:,1]
    auc_val = roc_auc_score(y_val, y_pred_val)
    print("AUC for fold ", n_fold, ": ", auc_val)
    auc_vals.append(auc_val)
    
    y_pred_test = clf.predict_proba(test_df[columns])[:,1]
    test_preds[n_fold, :] = y_pred_test
    print("----------------")
```
k-fold를 이용해 데이터를 5개로 나눈 다음 각 케이스마다 test 데이터를 예측한 후 test_preds에 저장했다. 쉽게 표현하면 다음과 같다. 또한, CatBoost는 자동으로 파라미터를 튜닝해주기 때문에 optuna와 F-Fold를 이용해 튜닝을 하지 않았다.

*  Fold 1: a로 검증, b + c + d + e로 훈련 → test_df 예측
*  Fold 2: b로 검증, a + c + d + e로 훈련 → test_df 예측
*  Fold 3: c로 검증, a + b + d + e로 훈련 → test_df 예측
*  Fold 4: d로 검증, a + b + c + e로 훈련 → test_df 예측
*  Fold 5: e로 검증, a + b + c + d로 훈련 → test_df 예측
```
submission_df['Exited'] = y_pred
submission_df.head()
submission_df.to_csv("submission.csv",index=False)
```
이후 위에서 했던 방법과 같이 평균을 낸 y_pred를 제출 파일에 입력한 후 제출했다. 결과는 아래와 같다. 결과적으로 CatBosot가 가장 높은 점수를 얻을 수 있었다.
![image](https://github.com/user-attachments/assets/a4d74ce1-e6a5-4fc7-b409-b4732d90e911)

   #### 4. Data Leakage

이번 대회에서는 Data Leakage가 있었다. 물론 필자는 Data Leakage가 일어나지 않았지만 공유된 코드를 보면 높은 점수를 얻은 사람들의 대부분은 Data Leakage가 있었다.

kaggle에서 제공한 파일은 train, test, submission 이렇게 세 개의 파일이다. 하지만 공유된 코드를 보면 Original Data가 등장한다.

![image](https://github.com/user-attachments/assets/78ad183d-3e2d-4d1c-b364-78e23fd07886)

Original Data를 학습 데이터에 포함해서 아래와 같이 학습을 했다. 
```
df_train = pd.concat([df_train, original_data], axis=0)
```
![image](https://github.com/user-attachments/assets/bb5c87a8-724e-4048-965c-06656d563eff)

하지만 위의 사진과 같이 점수는 필자와 비슷하게 나왔다.

---

### 결론
이번 Kaggle 분류 대회를 통해 고객 이탈(Churn)을 예측하는 과정을 다루면서 다양한 모델과 기법을 실험했습니다. 데이터를 처리하는 과정에서 EDA(탐색적 데이터 분석)를 통해 주요한 패턴과 특성을 발견하였고, 이를 바탕으로 Feature Engineering을 통해 예측 성능을 높이기 위한 새로운 변수를 추가했습니다. 또한, LightGBM, XGBoost, CatBoost와 같은 다양한 부스팅 모델을 사용해 성능을 평가했으며, 각 모델의 장단점과 결과를 비교하는 과정을 거쳤습니다.

* Feature Engineering의 중요성
고객 이탈 예측에서 중요한 변수로는 나이, 신용 점수, 은행 서비스 이용 기간, 사용 중인 금융 상품의 수 등이 있었습니다. 이러한 변수들은 고객의 행동 패턴을 반영할 수 있으며, 이탈 가능성을 잘 예측할 수 있는 변수들로 드러났습니다. 특히, 나이를 연령대 그룹으로 나누고, 고객이 이용하는 상품 수와 이용 기간을 조합하여 만든 변수는 모델의 성능을 향상시키는 데 중요한 역할을 했습니다.
* 모델 비교 및 성능 평가
세 가지 대표적인 부스팅 모델(XGBoost, LightGBM, CatBoost)을 사용하여 모델을 학습하고 검증한 결과, CatBoost 모델이 가장 높은 AUC(Area Under the ROC Curve)를 기록했습니다. CatBoost는 범주형 변수를 자동으로 처리하고, 테스트 데이터에서 새로운 범주형 변수를 인식할 수 있어, 이 과정에서 특별한 인코딩 작업을 생략할 수 있는 강점이 있었습니다. 이로 인해 더 높은 일반화 성능을 발휘할 수 있었습니다. LightGBM과 XGBoost도 좋은 성능을 보였으나, 데이터셋의 특성상 CatBoost가 더 적합한 모델임을 확인할 수 있었습니다.
* 교차 검증을 통한 모델의 일반화
교차 검증(K-Fold)을 통해 모델의 성능을 평가한 결과, 과적합 없이 안정적인 성능을 보여주는 것이 중요함을 다시 한 번 확인했습니다. 각 Fold에서 비슷한 AUC 점수를 기록한 모델은 일반화된 성능을 가지고 있다는 것을 의미하며, 이것이 Kaggle 대회에서 중요한 요소 중 하나임을 느꼈습니다. 다양한 Fold에서 훈련과 검증을 거쳐 최종 예측값을 평균화하는 방식은, 개별 Fold에서 발생할 수 있는 불균형한 데이터를 보완해 줍니다.
* Data Leakage의 중요성
이번 대회에서는 Data Leakage가 일부 참가자들 사이에서 발생한 점이 눈에 띄었습니다. 본래 Kaggle 대회에서는 제공된 학습 데이터와 테스트 데이터만을 사용해야 하지만, 일부 참가자들은 공유된 'Original Data'를 학습에 포함하여 점수를 높였습니다. Data Leakage는 모델의 성능을 부정확하게 높일 수 있으며, 실제 환경에서는 신뢰할 수 없는 결과를 초래할 수 있기 때문에 주의해야 할 요소입니다. 필자는 이러한 점을 인지하고, 데이터 누출이 없는 방식으로 모델을 구축하여, 공정하게 성능을 평가할 수 있었습니다.

#### 한계점
* 불균형 데이터
고객 이탈 예측에서 타겟 클래스(Exited = 1)와 비이탈 클래스(Exited = 0)의 불균형은 주요한 문제였습니다. 이로 인해 모델이 이탈하지 않는 고객을 과대 평가할 가능성이 있었고, 이는 F1 점수에서 낮은 재현율로 나타났습니다. 이러한 문제를 해결하기 위해 언더샘플링, 오버샘플링, SMOTE와 같은 기법을 사용해 데이터 불균형을 해결할 수도 있었으나, 이번 대회에서는 이러한 기법을 활용하지 않았습니다. 향후에는 더 복잡한 기법을 적용하여 불균형 문제를 해결할 수 있을 것입니다.
* 데이터 이해 부족
제공된 데이터는 실제 금융 데이터가 아니었기 때문에 실제 은행에서 고객 이탈을 예측하는 문제와는 다를 수 있습니다. 이로 인해, 일부 변수의 패턴이 실제 금융 데이터에서는 다르게 나타날 수 있습니다. 또한, 이번 프로젝트에서는 피처에 대한 더 깊은 도메인 지식을 바탕으로 한 변형이 제한적이었으므로, 실제 적용에서는 추가적인 변형 및 도메인 지식이 필요할 것입니다.

#### 마무리
이번 프로젝트를 통해 분류 문제에서 데이터 전처리, Feature Engineering, 모델 학습, 그리고 성능 평가에 대한 종합적인 경험을 쌓을 수 있었습니다. 특히, LightGBM, XGBoost, CatBoost와 같은 부스팅 모델의 강력함을 확인할 수 있었고, 데이터 처리와 모델 선택이 성능에 미치는 영향을 경험할 수 있었습니다. 분류 문제는 데이터의 특성과 모델의 선택이 중요한 역할을 하며, 이를 잘 이해하고 활용하는 것이 좋은 성능을 이끌어 내는 핵심임을 깨달았습니다.

다음 게시물부터는 회귀 문제를 다루며 새로운 인사이트를 얻는 과정을 진행하려고 합니다. 이번 분류 문제를 통해 얻은 경험을 바탕으로 더 발전된 회귀 분석을 다룰 예정이며, 이와 같은 과정을 반복하며 데이터 분석 및 머신러닝 모델링 역량을 지속적으로 향상시킬 것입니다.


