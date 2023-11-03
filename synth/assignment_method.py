# 各タスクの候補ワーカーのAの値平均を求める
# A: 各ワーカーの候補リスト中の登場回数
import copy
import numpy as np

# アルゴリズム
  
def flatten(assign_dic, worker_list, i):
  worker = worker_list[i]
  pre_worker =  worker_list[i-1]

  # if len(assign_dic[worker]) < len(assign_dic[pre_worker]):
  while len(assign_dic[worker]) < len(assign_dic[pre_worker]):
    # print(assign_dic[worker], assign_dic[pre_worker], i)
    task = assign_dic[pre_worker][0]
    assign_dic[pre_worker].remove(task)
    assign_dic[worker].append(task)
    
    if i > 1:
      flatten(assign_dic, worker_list, i-1)

    #print('====-!====!====!=====')
    #print(assign_dic[worker], assign_dic[pre_worker], i)
 
  
 
  return assign_dic

def optim_assignment(worker_c, test_worker, test_task, user_param):
  # (1) ワーカを能力が低い順にソート
  worker_list = []
  remain_task = []
  for task in test_task:
    remain_task.append(task)

  user_param_sorted = dict(sorted(user_param.items(), key=lambda x: x[1]))
  for worker in user_param_sorted.keys():
    if worker in test_worker:
      worker_list.append(worker)

  # worker: assigned(w)
  assign_dic = {}
  for worker in worker_list:
    assign_dic[worker] = []
  

  n = len(worker_list)
  # w_1, ... , w_nについて
  for i in range(0, n):
    worker = worker_list[i]
    # 可能なタスクをすべて割当て
    for task in worker_c:
      # print(worker, worker_c[task])
      if (worker in worker_c[task] and task in remain_task):
        assign_dic[worker].append(task)
        remain_task.remove(task)
    if i >= 1:
      pre_worker = worker_list[i-1]
      # i-1のワーカよりassignedが小さければ
      assign_dic = flatten(assign_dic, worker_list, i)
    
  return assign_dic



# 各テストタスクの正解率平均求める
def accuracy(assign_dic, input_df):
  score = 0
  task_num = 0
  for task in assign_dic:
    worker = assign_dic[task]
    task_num += 1
    #print(task, worker)
    if input_df[worker][task] == 1:
      score += 1
  # print(task_num)
  # print(score, task_num)
  if task_num > 0:
    acc = score/task_num
  # assign_dicが空の場合
  else:
    acc = "null"
  return acc

# 各ワーカーの割り当てタスク数数える
def task_variance(assign_dic, test_workers):
  count_dic = {}
  for tw in test_workers:
    count_dic[tw] = 0
  for aw in assign_dic.values():
    count_dic[aw] += 1

  # 担当タスク数の分散を数える
  count_list = list(count_dic.values())

  v = np.var(count_list)
  return v

def calc_tp(assign_dic, test_worker):
  count_dic = {}
  # print(f'assign_dic: {assign_dic}')
  for tw in test_worker:
    assign_worker_list = list(assign_dic.values())
    workload = assign_worker_list.count(tw)
    count_dic[tw] = workload

  # 最大のvalueを調べる
  max_num =  max(list(count_dic.values()))
  return max_num