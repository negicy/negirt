import matplotlib.pyplot as plt
from irt_method import *
import sys, os
from assignment_method import *
import numpy as np
import pandas as pd
import girth

def task_survey(item_param, user_param, input_df):
    # タスクごとの各変数
    avg_task_score = {}
    for i in item_param:
        score = 0
        prob_sum = 0
        theta_sum = 0
        u_num = 0
        avg_task_score[i] = []
        # a = a_list[0]
        b = item_param[i]
        
        # avg_task_score[i].append(a)
        avg_task_score[i].append(b)
        for u in user_param:
            theta = user_param[u]
            theta_sum += theta

            u_num += 1
            if input_df[u][i] == 1:
                score += 1
            prob_sum += OnePLM(b, theta)
        #avg_theta = theta_sum / u_num
        # avg_task_score[i].append(avg_theta)
        avg_task_score[i].append(score/u_num)
        avg_task_score[i].append(prob_sum/u_num)

    print(avg_task_score)
    x = []
    y = []
    for l in avg_task_score.values():
        print(l)
        x.append(l[2])
        y.append(l[1])
        print(x)
        print(y)

        # タスクの難易度と平均正解率の散布図

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y)
    plt.show()
    # return avg_task_score




# すべてのタスクの平均正解率
def task_correct_rate(input_df, worker_list, task_list):
  correct_rate_dic = {}
  for i in task_list:
    correct = 0
    worker_num = 0
    for w in worker_list:
      worker_num += 1
      if input_df[w][i] == 1:
        correct += 1
    correct_rate_dic[i] = (correct / worker_num)
  return correct_rate_dic

def worker_correct_rate(input_df, worker_list, task_list):
  # 各ワーカーの全体正解率
  skill_rate_dic = {}
  # 各ワーカーの全体正解率
  skill_rate_dic = {}
  for w in worker_list:
    correct = 0
    task_num = 0
    for i in task_list:
      task_num += 1
      if input_df[w][i] == 1:
        correct += 1
    skill_rate_dic[w] = (correct / task_num)

  return skill_rate_dic


'''

# 本当の正答確率とタスクの平均正解率
th = 0.8
acc_count = 0
case = 0
prob_list = []
beta_list = []
cr_list = []

for item in task_list:
  prob_sum = 0
  beta = item_param[item]
  for worker in worker_list:
    theta = user_param[worker]
    prob = OnePLM(beta, theta)
    prob_sum += prob

  prob_mean = prob_sum / len(worker_list)
  prob_list.append(prob_mean)
  cr_list.append(correct_rate_dic[item])

fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot(1, 1, 1)

ax1.set_xlabel("average P among workers")
ax1.set_ylabel("correct answer rate from workers")

clrs = ['b', 'orange'] 
# x = np.array(px)
ax1.scatter(prob_list, cr_list, marker='.')

# plt.savefig("simulation-result-12.eps")
plt.show()

# 本当の正答確率とワーカーの平均正解率
th = 0.8
acc_count = 0
case = 0
prob_list = []
beta_list = []
cr_list = []

for worker in worker_list:
  prob_sum = 0
  theta = user_param[worker]
  for item in task_list:
    beta = item_param[item]
    prob = OnePLM(beta, theta)
    if prob > th:
      case += 1
      if input_df[worker][item] == 1:
        acc_count += 1
    prob_sum += prob
  prob_mean = prob_sum / len(task_list)
  prob_list.append(prob_mean)
  cr_list.append(skill_rate_dic[worker])

fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot(1, 1, 1)

ax1.set_xlabel("average P among workers")
ax1.set_ylabel("average correct answer rate of workers")
clrs = ['b', 'orange'] 

# x = np.array(px)
ax1.scatter(prob_list, cr_list, marker='.')

# plt.savefig("simulation-result-12.eps")
plt.show()

for i in item_param:
  score = 0
  prob_sum = 0
  theta_sum = 0
  u_num = 0
  avg_task_score[i] = []
  # a = a_list[0]
  b = item_param[i]
  
  # avg_task_score[i].append(a)
  avg_task_score[i].append(b)
  for u in user_param:
    theta = user_param[u]
    theta_sum += theta

    u_num += 1
    if input_df[u][i] == 1:
      score += 1
    prob_sum += OnePLM(b, theta)
  # avg_theta = theta_sum / u_num
  # avg_task_score[i].append(avg_theta)
  # タスクiの, ワーカーによる正解率平均
  avg_task_score[i].append(score/u_num)
  # タスクiの, ワーカーの正答確率の平均
  avg_task_score[i].append(prob_sum/u_num)

print(avg_task_score)
x = []
y = []
for l in avg_task_score.values():
  print(l)
  x.append(l[0])
  y.append(l[1])
print(x)
print(y)

'''