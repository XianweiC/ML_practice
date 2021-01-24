# -*- coding =utf-8 -*-
# @Time : 2021/1/22 14:41
# @Author : 曹显伟
# @File model.py.py
# @Software: PyCharm

from sklearn import datasets
import numpy as np

'''清空sklearn环境下所有数据'''
datasets.clear_data_home()

'''是sklearn包自带的数据集，
这个数据集包含了威斯康辛州记录的569个病人的乳腺癌恶性/良性（1/0）类别型数据（训练目标），
以及与之对应的30个维度的生理指标数据；
因此这是个非常标准的二类判别数据集，'''
X, y = datasets.load_breast_cancer(return_X_y = 'true')
# 将y(1,0)调整为(1,-1)
y = np.where(y==0, -1, 1)

'''
print(X.shape)
print(len(X))
print(y.shape)
'''


# 模型
class My_Perceptron():
    def __init__(self):
        self.W = np.ones((len(X[0]),), dtype=float)
        self.b = 0
        self.lr = 0.01
        # 先训练一百次看看效果
        self.epoch = 100

    def fit(self, X, y):
        for ep in range(self.epoch):
            for i in range(len(X)):
                # 异号说明是误分类点
                if y[i] * (np.dot(X[i], self.W) + self.b) < 0:
                    self.W += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]

    # 预测
    def predict(self, X):
        return np.where((np.dot(X, self.W) + self.b) > 0, 1, -1)

    # 评估
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.count_nonzero(y - y_pred) / len(y)

# 可以对模型进行验证看看
if __name__ == '__main__':
    perceptron = My_Perceptron()
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X)
    print(perceptron.score(X, y))