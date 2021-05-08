# -*- coding =utf-8 -*-
# @Time : 2021/4/4 12:12
# @Author : 曹显伟
# @File MyKNN.py
# @Software: PyCharm

import numpy as np

#  参考书上P197
_label = ['x', 'y']
_dataset = np.array([
    [0.5, '-'],
    [3.0, '+'],
    [4.5, '+'],
    [4.6, '+'],
    [4.9, '+'],
    [5.2, '-'],
    [5.3, '-'],
    [5.5, '+'],
    [7.0, '-'],
    [9.5, '-']
])


class MyKNN(object):

    def __init__(self, point, num):
        self.pre_point = point
        self.num_neighbors = num

    def get_point_list(self, dataset):
        point_list = np.array([dataset[i][0] for i in range(len(dataset))])
        return point_list

    # 计算两点之间距离
    def calc_distance(self, point1, point2):
        distance = abs(float(point1) ** 2 - float(point2) ** 2)
        return distance

    # 计算每个样本距离待预测点之间的距离，并排序后返回索引
    def get_dis_index(self):
        point_list = self.get_point_list(_dataset)
        distance_list = np.array([self.calc_distance(self.pre_point, i) for i in point_list])
        # 返回距离排序后的索引
        index = np.argsort(distance_list)

        return index

    def drow_conclusion(self):
        res_class = ''
        distance_index = self.get_dis_index()
        data = _dataset
        dict_re = {'+': 0, '-': 0}
        # 判断近邻出现的概率大小
        for i in range(self.num_neighbors):
            dict_re[data[i][1]] += 1
        # print(dict_re)
        maxs = 0
        for i in dict_re.values():
            if i >= maxs:
                maxs = i
        for key in dict_re.keys():
            if dict_re[key] == maxs:
                res_class = key
        return res_class


if __name__ == '__main__':
    pre_point = 5.0
    num_neighbor = 4
    knn = MyKNN(pre_point, num_neighbor)
    pre_result = knn.drow_conclusion()
    print(pre_result)
