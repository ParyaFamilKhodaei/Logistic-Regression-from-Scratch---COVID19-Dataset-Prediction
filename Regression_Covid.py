
# ***PART 1***

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('/content/covid.csv')
le=LabelEncoder()
print(df.head(10))

df = df.drop('entry_date',axis = 1).drop('date_symptoms', axis = 1)
df = df.dropna()
df = df[~df.isin(['#']).any(axis=1)]
categorical = [
    'sex',
    'patient_type',
    'intubed',
    'pneumonia',
    'pregnancy',
    'diabetes',
    'copd',
    'asthma',
    'inmsupr',
    'hypertension',
    'other_disease',
    'cardiovascular',
    'obesity',
    'renal_chronic',
    'tobacco',
    'contact_other_covid',
    'icu'
]
for col in categorical:
    df[col] = le.fit_transform(df[col].astype(str))
print(df)
df['date_died'] = (df['date_died'] != '9999-99-99').astype(int)

df_train=df.sample(frac=0.8)
df_test=df.drop(df_train.index)

x_train = df_train[['sex', 'patient_type', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']].values
y_train = df_train['date_died'].values
x_test = df_test[['sex', 'patient_type', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']].values
y_test = df_test['date_died'].values

train_min = x_train.min(axis=0)
train_max = x_train.max(axis=0)
denom = train_max - train_min
denom[denom == 0] = 1

x_train = (x_train - train_min) / denom
x_test  = (x_test  - train_min) / denom

iters = 3000
alpha = 0.1
b1=0
w1=np.zeros(x_train.shape[1])


def sigmoid(s):
  return 1 / (1 + np.exp(-s))


def cost(X, y, w, b):
    epsi = 1e-7
    m = X.shape[0]
    g = sigmoid((np.dot(X, w) + b))
    return (-(1/m) * np.sum(y * np.log(g +epsi) + (1 -g) * np.log(1-g+epsi)))


def comp_grad(X, y, w, b):
    m = X.shape[0]
    g = sigmoid(np.dot(X, w) + b)
    dwj = (1 / m) * np.dot(X.T, (g - y))
    dbj = (1 / m) * np.sum(g - y)
    return dwj, dbj


def grad_des(x,y,w,b,iters,alpha):
  cost_hist =[]
  for i in range(iters):
    D_w , d_b = comp_grad(x,y,w,b)
    w -= alpha*D_w
    b = b - d_b * alpha
    cost_ = cost(x,y,w,b)
    cost_hist.append(cost_)
    if i%100 == 0:
      print(f'cost at {i} is {cost_}')
  return w , b , cost_hist


print(cost(x_train,y_train,w1,b1))

w_train , b_train , cost_hist1 = grad_des(x_train,y_train,w1,b1,iters,alpha)

y_test0= y_test==0
y_test1= y_test==1

train_cost = cost(x_train, y_train, w_train, b_train)
test_cost  = cost(x_test, y_test, w_train, b_train)

print("Training Cost:", train_cost)
print("Testing Cost :", test_cost)

plt.plot(cost_hist1)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

def predict(X, w, b):
    f = np.dot(X, w) + b
    g = sigmoid(f)
    return np.where(g > 0.5, 1, 0)  #Threshold is 0.5


y_pred = predict(x_test, w_train, b_train)
y_pred0= y_pred == 0
y_pred1= y_pred == 1

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy
accuracy = accuracy_score(y_test0, y_pred0)
print("Accuracy:", accuracy,'for 0')
# Precision
precision = precision_score(y_test0, y_pred0)
print("Precision:", precision,'for 0')
# Recall
recall = recall_score(y_test0, y_pred0)
print("Recall:", recall,'for 0')
# F1score
f1 = f1_score(y_test0, y_pred0)
print("F1 score:", f1,'for 0')


# Accuracy
accuracy = accuracy_score(y_test1, y_pred1)
print("Accuracy:", accuracy,'for 1')
# Precision
precision = precision_score(y_test1, y_pred1)
print("Precision:", precision,'for 1')
# Recall
recall = recall_score(y_test1, y_pred1)
print("Recall:", recall,'for 1')
# F1score
f1 = f1_score(y_test1, y_pred1)
print("F1 score:", f1,'for 1')

plt.figure(figsize=(6,4))
plt.plot(cost_hist1)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Training Cost")
plt.grid(True)
plt.show()

# ***PART 2***

from google.colab import drive
drive.mount('/content/drive')

dtf = pd.read_csv('/content/drive/MyDrive/balanced_dataset_covid.csv')
dtf = dtf.drop('entry_date', axis = 1).drop('date_symptoms', axis = 1)
dtf = dtf.dropna()
dtf = dtf[~dtf.isin(['#']).any(axis=1)]
dtf['date_died'] = (dtf['date_died'] != '9999-99-99').astype(int)
for col in categorical:
    dtf[col] = le.fit_transform(dtf[col].astype(str))

print(dtf)

dtf_train=dtf.sample(frac=0.8)
dtf_test=dtf.drop(dtf_train.index)

xx_train = dtf_train[['sex', 'patient_type', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']].values
yy_train = dtf_train['date_died'].values
xx_test = dtf_test[['sex', 'patient_type', 'intubed', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', 'icu']].values
yy_test = dtf_test['date_died'].values

train_min = xx_train.min(axis=0)
train_max = xx_train.max(axis=0)
denom = train_max - train_min
denom[denom == 0] = 1

xx_train = (xx_train - train_min) / denom
xx_test  = (xx_test  - train_min) / denom

w2=np.zeros(xx_train.shape[1])
b2=0
iters=8000

print(cost(xx_train,yy_train,w2,b2))

wn, bn, hist = grad_des(xx_train, yy_train, w2, b2, iters, alpha)

train_cost = cost(xx_train, yy_train, wn, bn)
test_cost = cost(xx_test, yy_test, wn, bn)
print("Training Cost:", train_cost)
print("Testing Cost:", test_cost)

yy_pred = predict(xx_test, wn, bn)


print("Accuracy :", accuracy_score(yy_test, yy_pred))
print("Precision:", precision_score(yy_test, yy_pred))
print("Recall   :", recall_score(yy_test, yy_pred))
print("F1 Score :", f1_score(yy_test, yy_pred))

print(f'Last train cost: {hist[-1]}')
print(f'Test cost: {test_cost}')

plt.figure(figsize=(6,4))
plt.plot(hist)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Balanced Dataset Training Cost")
plt.grid(True)
plt.show()
