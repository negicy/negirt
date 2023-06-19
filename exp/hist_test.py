import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 目盛り(x方向)とランダム値(y方向)を生成
x1 = [i for i in range(20)]
x2 = [0.5, 0.6, 0.7, 0.8]
y1 = [np.random.randint(0, 100) for i in range(20)]

# 2つ目のデータを作成
y2 = [np.random.randint(0, 100) for i in range(20)]

fig, ax = plt.subplots(2, sharey=True)

# 横に並べて描画
# x座標は横に並べるために位置をシフト
ax[0].bar(x1, y1, width=0.4)
ax[0].bar(np.array(x1) + 0.4, y2, width=0.4)

# y1の高さをbottomとして2つめのデータを描画
ax[1].bar(x1, y1, width=0.4)
ax[1].bar(x1, y2, bottom=y1, width=0.4)

# グラフタイトルの設定
ax[0].set_title('default')
ax[1].set_title('stack')

fig.tight_layout()
print(x1)
print(y1)
print(y2)
plt.show()
