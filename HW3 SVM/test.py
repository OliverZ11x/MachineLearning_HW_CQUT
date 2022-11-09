import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import train_test_split


raw_data = loadmat('F:/MachineLearning/MachineLearning_HW_CQUT-master/HW3 SVM/data2.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']


data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)#训练集：测试集=7：3

svc = svm.SVC(kernel='poly', C=100, gamma='auto')
svc.fit(data_train[['X1', 'X2']], data_train['y'])
svc.score(data_test[['X1', 'X2']], data_test['y'])



positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=25, marker='o', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=25, marker='x', label='Negative')
ax.legend()


# 决策边界, 使用等高线表示
x1 = np.arange(0.04, 1, 0.01)
x2 = np.arange(0.4, 1, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)

plt.show()

