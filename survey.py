import matplotlib.pyplot as plt
from irt_method import *
import sys, os
from assignment_method import *
import numpy as np
import pandas as pd
import girth

# タスクの全体正解率
def task_correct_rate(task_list, worker_list, input_df):
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

# 各ワーカーの全体正解率
def worker_correct_rate(task_list, worker_list, input_df):
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

def Frequency_Distribution(data, lim, class_width=None):
    data = np.asarray(data)
    if class_width is None:
        class_size = int(np.log2(data.size).round()) + 1
        class_width = round((data.max() - data.min()) / class_size)

    bins = np.arange(lim[0], lim[1], class_width)
    print(bins)
    hist = np.histogram(data, bins)[0]
    cumsum = hist.cumsum()

    return pd.DataFrame({'階級値': (bins[1:] + bins[:-1]) / 2,
                         '度数': hist,
                         '累積度数': cumsum,
                         '相対度数': hist / cumsum[-1],
                         '累積相対度数': cumsum / cumsum[-1]},
                        index=pd.Index([f'{bins[i]}以上{bins[i+1]}未満'
                                        for i in range(hist.size)],
                                       name='階級'))


label_df = pd.read_csv("label_df.csv", sep = ",")
input_df = pd.read_csv("input.csv", sep = ",")
# label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
# label_df = label_df.set_index('id')

batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
label_df = label_df.set_index('id')

# origin_id: max(prob_dic)のdict作成
task_dic = {}
for k in range(0, 100):
  task_id = "q"+str(k+1)
  # 元のテストデータのIDを取り出す
  origin_index = 'Input.'+task_id
  origin_id = batch_df[origin_index][0]
  task_dic["q"+str(k+1)] = origin_id

print(task_dic)
input_df = input_df.set_index('qid')
input_df['task_id'] = 0

q_list = list(input_df.index)
print(q_list)

# Task IDリストの作成
task_list = list(task_dic.values())
print(task_list)

# input_dfのインデックスを置き換え
for q in q_list:
  input_df['task_id'][q] = task_dic[q]
input_df = input_df.set_index('task_id')

worker_list = list(input_df.columns)
# ワーカーリスト作成~割り当て　実行

# すべてのタスクの平均正解率
correct_rate_dic = {}
for i in task_list:
  correct = 0
  worker_num = 0
  for w in worker_list:
    worker_num += 1
    if input_df[w][i] == 1:
      correct += 1
  correct_rate_dic[i] = (correct / worker_num)
print(correct_rate_dic)

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

print(skill_rate_dic)

acc_all_th = []
var_all_th = []

df = pd.read_csv('input.csv')
df = df.set_index('qid')
# print(df)
#task_list = list(df.index)
#worker_list = list(df.columns)
threshold = list([i / 100 for i in range(50, 81)])
# data: 0/1 のndarray (2d)  
data = df.values

params = run_girth_onepl(data, task_list, worker_list)

# params
item_param = params[0]
user_param = params[1]

# 各タスクの正答率求める
# タスクごとの各変数
avg_task_score = {}

# task_survey(item_param, user_param, input_df)

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
  #avg_theta = theta_sum / u_num
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

# タスクの難易度と平均正解率の散布図

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('task difficulty')
ax.set_ylabel('average P among workers')
ax.scatter(x, y)
plt.show()


# 推定難易度と実際の難易度の相関
  