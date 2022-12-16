import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import girth
import random
import scipy
import scikit_posthocs as sp
from assignment_method import *
from irt_method import *
from simulation import *
from survey import *


import pickle


filename = 'result_20220630_210655.txt.pkl'

with open(filename, 'rb') as p:
    results = pickle.load(p)

welldone_dist = results['welldone_dist']

result = {
  'ours_output': ours_output_alliter, 'full_output': full_output_alliter, 
  'ours_acc': ours_acc_allth, 'top_acc': top_acc_allth, 
  'random_acc': random_acc_allth, 'PI_acc': PI_acc_allth,
  'ours_var': ours_var_allth, 'top_var': top_var_allth, 
  'random_var': random_var_allth, 'PI_var': PI_var_allth,
  'welldone_dist': welldone_dist, 
  #'ours_acc_head': ours_acc_head, 'AA_acc_head': AA_acc_head,
  #'ours_acc_tail': ours_acc_tail, 'AA_acc_tail': AA_acc_tail
}



ours_acc = results['ours_acc_allth']


iteration_time = 5

for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time

plt.rcParams["font.size"] = 18
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('number of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.4)
plt.show()


# 割当て成功数ヒストグラム
plt.rcParams["font.size"] = 18
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.5, color='forestgreen')
plt.show()


# タスクのあるワーカ人数をヒストグラムで
# iteration間の平均を求める



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
  'ours': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'full_irt': full_irt_acc,
  'ours_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'full_irt_std': full_acc_std
  }

result_var_dic = {
  'ours': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'full_irt': full_irt_var,
  'ours_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'full_irt_std': full_var_std
}

result_plot_1(threshold, result_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_1(threshold, result_var_dic, ay='variance', bbox=(0.150, 0.800)).show()

# トレードオフのグラフ
ours_trade = tp_acc_plot(ours_tp, ours_acc)
AA_trade = tp_acc_plot(AA_tp, AA_acc)
top_trade = tp_acc_plot(top_tp, top_acc)
random_trade = tp_acc_plot(random_tp, random_acc)
PI_trade = tp_acc_plot(PI_tp, full_irt_acc)
PI_noise1_trade = tp_acc_plot(PI_noise1_tp, PI_noise1_acc)

# top_trade = var_acc_plot(top_var, top_acc)
# random_trade = var_acc_plot(random_var, random_acc)

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


# 推移をプロット
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('threshold')
ax.set_ylabel('margin')
x = np.array(threshold)

ax.plot(x, DI_margin_result, color='red', label='IRT(DI)')
ax.plot(x, PI_margin_result, color='purple', label='IRT(PI)')
plt.show()





