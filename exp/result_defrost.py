import pickle
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy
import scikit_posthocs as sp
from assignment_method import *
from irt_method import *
from simulation import *
from make_candidate import *

filename = 'results/result_20230719_103944.pickle'

with open(filename, 'rb') as p:
    results = pickle.load(p)


welldone_dist = results['welldone_dist']
iteration_time = 200

ours_acc = results['ours_acc']
ours_var = results['ours_var']
ours_tp = results['ours_tp']
ours_acc_std = results['ours_acc_std']
ours_var_std = results['ours_var_std']

top_acc = results['top_acc']
top_var = results['top_var']
top_tp = results['top_tp']
top_acc_std = results['top_acc_std']
top_var_std = results['top_var_std']

AA_acc = results['AA_acc']
AA_var = results['AA_var']
AA_tp = results['AA_tp']
AA_acc_std = results['AA_acc_std']
AA_var_std = results['AA_var_std']

random_acc = results['random_acc']
random_var = results['random_var']
random_tp = results['random_tp']
random_acc_std = results['random_acc_std']
random_var_std = results['random_var_std']


PI_acc = results['PI_acc']
PI_var = results['PI_var']
PI_tp = results['PI_tp']
PI_acc_std = results['PI_acc_std']
PI_var_std = results['PI_var_std']

threshold = list([i / 100 for i in range(50, 81)])
threshold =  [0.5, 0.6, 0.7, 0.8]


for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time

plt.rcParams["font.size"] = 22
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.4)
plt.show()

# タスクのあるワーカのヒストグラム
worker_with_task = results['worker_with_task']

num_worker = [[], []]
for th in [0.5, 0.6, 0.7, 0.8]:
  num_worker[0].append(worker_with_task['ours'][th] / iteration_time)
  num_worker[1].append(worker_with_task['AA'][th] / iteration_time)
w = 0.4
y_ours = num_worker[0]
y_AA = num_worker[1]


x1 = [1, 2, 3, 4]
x2 = [1.3, 2.3, 3.3, 4.3]

# 少なくとも1つ以上のタスクを与えられたワーカのヒストグラム
label_x = ['0.5', '0.6', '0.7', '0.8']
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
# 1つ目の棒グラフ
plt.bar(x1, y_ours, color='blue', width=0.3, label='DI', align="center")

# 2つ目の棒グラフ
plt.bar(x2, y_AA, color='coral', width=0.3, label='AA', align="center")

# 凡例
plt.xlabel('threshold')
plt.ylabel('Number of workers with tasks')
# X軸の目盛りを置換
plt.xticks([1.15, 2.15, 3.15, 4.15], label_x)
fig.legend(bbox_to_anchor=(0.15, 0.250), loc='upper left')
plt.show()



# 推移をプロット

result_acc_dic = {
  'DI': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_acc,
  'DI_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_acc_std

  }

result_var_dic = {
  'DI': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_var,
  'DI_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_var_std
}

result_plot_acc_var(threshold, result_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_acc_var(threshold, result_var_dic, ay='variance', bbox=(0.150, 0.800)).show()

# トレードオフのグラフ
ours_trade = tp_acc_plot(ours_tp, ours_acc)
AA_trade = tp_acc_plot(AA_tp, AA_acc)
top_trade = tp_acc_plot(top_tp, top_acc)
random_trade = tp_acc_plot(random_tp, random_acc)
PI_trade = tp_acc_plot(PI_tp, PI_acc)

# top_trade = var_acc_plot(top_var, top_acc)
# random_trade = var_acc_plot(random_var, random_acc)

plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('Working Opportunity')
ax.set_ylabel('accuracy')
ax.set_xlim(0, 55)

bbox=(0.4750, 0.400)
ax.plot(ours_trade[0], ours_trade[1], color='red', label='IRT')
ax.plot(AA_trade[0], AA_trade[1], color='cyan', label='AA')
ax.plot(top_trade[0], top_trade[1], color='blue', label='TOP')
ax.plot(random_trade[0], random_trade[1], color='green', label='RANDOM')
ax.plot(PI_trade[0], PI_trade[1], color='purple', label='IRT(PI)')
fig.legend(bbox_to_anchor=bbox, loc='upper left')
plt.show()


# トレードオフのグラフ
ours_trade = var_acc_plot(ours_var, ours_acc)
AA_trade = var_acc_plot(AA_var, AA_acc)
top_trade = var_acc_plot(top_var, top_acc)
random_trade = var_acc_plot(random_var, random_acc)
PI_trade = var_acc_plot(PI_var, PI_acc)


plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('Working Opportunity')
ax1.set_ylabel('accuracy')
ax1.set_xlim(0, 30)

bbox=(0.2750, 0.400)
ax1.plot(ours_trade[0], ours_trade[1], color='red', label='IRT')
ax1.plot(AA_trade[0], AA_trade[1], color='cyan', label='AA')
ax1.plot(top_trade[0], top_trade[1], color='blue', label='TOP')
ax1.plot(random_trade[0], random_trade[1], color='green', label='RANDOM')
ax1.plot(PI_trade[0], PI_trade[1], color='purple', label='IRT(PI)')
# ax1.plot(PI_noise1_trade[0], PI_noise1_trade[1], color='orange', label='IRT(PI0.5)')
fig.legend(bbox_to_anchor=bbox, loc='upper left')
plt.show()


result_acc_dic = {
  'DI': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_acc,
  'DI_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_acc_std
  }

result_var_dic = {
  'DI': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_var,
  'DI_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_var_std
}

result_tp_dic = {
  'DI': ours_tp, 'top': top_tp, 'AA': AA_tp, 'random': random_tp, 'PI': PI_tp
}


#result_plot_tradeoff(result_tp_dic, result_acc_dic).show()
result_plot_tradeoff(result_var_dic, result_acc_dic).show()