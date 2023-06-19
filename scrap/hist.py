import matplotlib.pyplot as plt

'''
y_ours = [1, 2, 3, 4]
y_AA = [10, 10, 10, 10]

x = ['0.5', '0.6', '0.7', '0.8']
plt.bar(x, y_ours, width=0.3, label='DI', align="center")
plt.bar(x, y_AA, width=0.3, label='AA', align="center")
plt.legend(loc=2)
plt.xticks(x, x)
# plt.show()


x1 = [1, 2, 3]
y1 = [4, 5, 6]

x2 = [1.3, 2.3, 3.3]
y2 = [2, 4, 1]

y_ours = [1, 2, 3, 4]
y_AA = [10, 10, 10, 10]

label_x = ['0.5', '0.6', '0.7', '0.8']
# label_x = ['Result1', 'Result2', 'Result3']

# 1つ目の棒グラフ
plt.bar(label_x, y_ours, color='b', width=0.3, label='Data1', align="center")

# 2つ目の棒グラフ
plt.bar(label_x, y_AA, color='g', width=0.3, label='Data2', align="center")

# 凡例
plt.legend(loc=2)

# X軸の目盛りを置換
plt.xticks([1.15, 2.15, 3.15, 4.15], label_x)
plt.show()
'''

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x1 = [1, 2, 3, 4]
y_ours = [1, 2, 3, 4]


x2 = [1.3, 2.3, 3.3, 4.3]
y_AA = [10, 10, 10, 10]

label_x = ['0.5', '0.6', '0.7', '0.8']

# 1つ目の棒グラフ
plt.bar(x1, y_ours, color='b', width=0.3, label='Data1', align="center")

# 2つ目の棒グラフ
plt.bar(x2, y_AA, color='g', width=0.3, label='Data2', align="center")

# 凡例
plt.legend(loc=2)

# X軸の目盛りを置換
plt.xticks([1.15, 2.15, 3.15, 4.15], label_x)
plt.show()