# -*- coding =utf-8 -*-
# @Project : DataMining
# @Time : 20:26
# @Author : XianweiCao
# @Package :
# @File : model.py
# @Software: PyCharm Professional

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
# 导入关联规则包
from mlxtend.frequent_patterns import association_rules

# 设置数据集
# from numpy.core import records

records = [['牛奶', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
           ['莳萝', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
           ['牛奶', '苹果', '芸豆', '鸡蛋'],
           ['牛奶', '独角兽', '玉米', '芸豆', '酸奶'],
           ['玉米', '洋葱', '洋葱', '芸豆', '冰淇淋', '鸡蛋']]

te = TransactionEncoder()
# 进行 one-hot 编码
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
# 利用 Apriori 找出频繁项集
freq = apriori(df, min_support=0.05, use_colnames=True)

# 计算关联规则
result = association_rules(freq, metric="confidence", min_threshold=0.6)
# 设置不省略显示
pd.set_option('display.max_rows', None)
result.sort_values(by="support")
print(result)

'''
association_rules(df, metric="confidence",
                      min_threshold=0.8,
                      support_only=False):

参数介绍：
- df：Apriori 计算后的频繁项集。
- metric：可选值['support','confidence','lift','leverage','conviction']。
里面比较常用的就是置信度和支持度。这个参数和下面的min_threshold参数配合使用。
- min_threshold：参数类型是浮点型，根据 metric 不同可选值有不同的范围，
    metric = 'support'  => 取值范围 [0,1]
    metric = 'confidence'  => 取值范围 [0,1]
    metric = 'lift'  => 取值范围 [0, inf]
support_only：默认是 False。仅计算有支持度的项集，若缺失支持度则用 NaNs 填充。
'''
