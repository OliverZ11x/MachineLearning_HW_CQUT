import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2

gaussian_kernel(x1, x2, sigma)

raw_data = loadmat('F:\MachineLearning\MachineLearning_HW_CQUT\HW3 SVM\data2.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']



positive = data_train[data_train['y'].isin([1])]
negative = data_train[data_train['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax.legend()
plt.show()



svc = svm.SVC(C=100, gamma=1, probability=True)
svc.fit(data_train[['X1', 'X2']], data_train['y'])

svc.score(data_test[['X1', 'X2']], data_test['y'])

data_test['Probability'] = svc.predict_proba(data_test[['X1', 'X2']])[:,0]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data_test['X1'], data_test['X2'], s=30, c=data_test['Probability'], cmap='Reds')


# 决策边界, 使用等高线表示
x1 = np.arange(0.04, 1, 0.01)
x2 = np.arange(0.4, 1, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)


plt.show()

