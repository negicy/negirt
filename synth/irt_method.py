import math
from tkinter import N
import girth
import pandas as pd
import numpy as np
import statistics
from girth import twopl_mml, onepl_mml, ability_mle, rasch_mml
import random

def TwoPLM(alpha, beta, theta, d=1.7, c=0.0):
  #p = 1 / (1+math.exp(-d*alpha*(theta-beta)))
  p = c + (1.0 - c) / (1 + math.exp(-(alpha * theta + beta)))
  return p
def OnePLM(beta, theta, d=1.7, c=0.0):
  p = 1 / (1+np.exp(-d*(theta-beta)))
  # p = c + (1.0 - c) / (1 + np.exp(-(1 * theta + beta)))
  return p
def TwoPLM_test(a, b, theta, d=1.7):
  p = 1 / (1+np.exp(-d*a*(theta-b)))
  return p
  


# input: 01の2D-ndarray, 対象タスクのリスト, 対象ワーカーのリスト


def run_girth_onepl(data, task_list, worker_list):
    estimates = onepl_mml(data)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']
    #print(discrimination_estimates)
    # print(difficulty_estimates)

    a_list = []
    for i in range(len(task_list)):
        a_list.append(discrimination_estimates)

    abilitiy_estimates = ability_mle(data, difficulty_estimates, a_list)
    print(abilitiy_estimates)

    user_param = {}
    item_param = {}

    for k in range(len(task_list)):
        task_id = task_list[k]

        item_param[task_id] = difficulty_estimates[k]
    for j in range(len(worker_list)):
        worker_id = worker_list[j]
        user_param[worker_id] = abilitiy_estimates[j] 

    # print(item_param)
    # print(len(user_param))
    return item_param, user_param

# input: 01の2D-ndarray, 対象タスクのリスト, 対象ワーカーのリスト
def run_girth_twopl(data, task_list, worker_list):
    estimates = twopl_mml(data)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']
    #print(discrimination_estimates)
    # print(difficulty_estimates)

    a_list = []
    for i in range(len(task_list)):
        a_list.append(discrimination_estimates)

    abilitiy_estimates = ability_mle(data, difficulty_estimates, a_list)
    # print(abilitiy_estimates)

    user_param = {}
    item_param = {}

    for k in range(len(task_list)):
        task_id = task_list[k]
        item_param[task_id] = {}
        item_param[task_id]['a'] = discrimination_estimates[k]
        item_param[task_id]['b'] = difficulty_estimates[k]
    for j in range(len(worker_list)):
        worker_id = worker_list[j]
        user_param[worker_id] = abilitiy_estimates[j] 

    # print(item_param)
    # print(len(user_param))
    return item_param, user_param

# qualification task, test taskに分けてシミュレーション
# IRT: girthを使用
# input: 01の2D-ndarray, 対象タスクのリスト, 対象ワーカーのリスト
def run_girth_rasch(data, task_list, worker_list):
    estimates = rasch_mml(data, 1)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']

    print(discrimination_estimates)
    #print(difficulty_estimates)

    abilitiy_estimates = ability_mle(data, difficulty_estimates, discrimination_estimates)
    # print(abilitiy_estimates)

    user_param = {}
    item_param = {}

    for k in range(len(task_list)):
        task_id = task_list[k]

        item_param[task_id] = difficulty_estimates[k]
    for j in range(len(worker_list)):
        worker_id = worker_list[j]
        user_param[worker_id] = abilitiy_estimates[j] 

    # print(item_param)
    # print(len(user_param))
    return item_param, user_param

# ワーカとタスクを分離
def devide_sample(task_list, worker_list):
  output = {}
  random.shuffle(task_list)
  n = 40
  qualify_task = task_list[:n]
  test_task = list(set(task_list) - set(qualify_task))
 
  output['qualify_task'] = qualify_task
  output['test_task'] = test_task
  output['test_worker'] = random.sample(worker_list, 20)

  return output
  
from scipy.stats import norm
import matplotlib.pyplot as plt 
import random

def ai_model(actual_b, dist):
    b_norm = norm.rvs(loc=actual_b, scale=dist, size=1000)
    #print(b_norm)
    ai_accuracy = list(b_norm).count(actual_b) / len(b_norm)
    #print(ai_accuracy)
    return random.choice(b_norm)


# 割当て候補のいないタスクを無くす
def sort_test_worker(test_worker, user_param, N=3):
  test_worker_param = {}
  for worker in test_worker:
    test_worker_param[worker] = user_param[worker]

  sorted_user_param = dict(sorted(test_worker_param.items(), key=lambda x: x[1], reverse=True))
  top_workers = list(sorted_user_param.keys())[:N]
  return top_workers

def make_candidate(threshold, input_df, worker_list, test_worker, qualify_task, test_task, full_item_param):
  worker_c_th = {}
  qualify_dic = {}
  qualify_dic = {}
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))

  # 各テストタスクについてワーカー候補を作成する
  # output: worker_c = {task: []}
  # すべてのスレッショルドについてワーカー候補作成

  params = run_girth_rasch(q_data, qualify_task, worker_list)
  est_item_param = params[0]
  est_user_param = params[1]
  DI_item_param = {}

  top_workers = sort_test_worker(test_worker, est_user_param)

  for th in threshold:
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # betaを推定する，

      for worker in test_worker:
        # workerの正答確率prob
        actual_b = full_item_param[task]
        ai_b = ai_model(actual_b, dist=0.01)
        DI_item_param[task] = ai_b
     
        theta = est_user_param[worker]
        prob = OnePLM(ai_b, theta)

        # workerの正解率がthresholdより大きければ
        if prob >= th:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)
      # 割当て候補リストが空の場合
      if len(worker_c[task]) == 0:
        worker_c[task] = top_workers
    
    worker_c_th[th] = worker_c

  return worker_c_th, est_user_param, DI_item_param

def task_assignable_check(th, item_param, user_param, test_worker, task):
  # sorted_full_user_param = list(sorted(full_user_param.items(), key=lambda x: x[1], reverse=True))
  # print(sorted_full_user_param)
 
  b = item_param[task]
  for worker in test_worker:
    # top_theta = sorted_full_user_param[0][1]
    theta = user_param[worker]
    prob = OnePLM(b, theta)
    if prob >= th:
      return True
  
  return False


def make_candidate_all(threshold, input_df, full_item_param, full_user_param, test_worker, test_task):
  worker_c_th = {}
  top_result = {}
 
  # print(item_param)
  # 難易度の最小値, 最大値
  theta_range = []
  #theta_range.append(np.min(list(full_item_param.values())))
  #theta_range.append(np.max(list(full_item_param.values())))  

  for th in threshold:
    # margin = th / 5
    # margin = th/20
    margin = 0
    worker_c = {}
    for task in test_task:
      if task_assignable_check(th+margin, full_item_param, full_user_param, test_worker, task) == True:
      
        # Aタスク
        worker_c[task] = []
        beta = full_item_param[task]
        for worker in test_worker:
          # workerの正答確率prob
          theta = full_user_param[worker]
          prob = OnePLM(beta, theta)
          # print(prob)
          # prob = TwoPLM(alpha, beta, theta, d=1.7)

          # workerの正解率がthresholdより大きければ
          if prob >= th + margin:
            # ワーカーを候補リストに代入
            worker_c[task].append(worker)
         
    worker_c_th[th] = worker_c

  return worker_c_th, full_item_param, full_user_param, top_result



# 平均正解率がthreshold以上のワーカに割り当てる
def entire_top_workers(threshold, input_df, test_worker, q_task, test_task):
  # test_workerについてqualificationの正解率を計算する from input_df
  # ワーカーのqualifivation taskの正解率を格納する辞書
  worker_rate = {}
  top_worker_dic = {}
  # test worker のtestタスクのqualification 正解率を調べる
  for worker in test_worker:
    q_score = 0
    # q_taskにおける平均正解率の計算
    for task in q_task:
      if input_df[worker][task] == 1:
        q_score += 1
    # 平均正解率の計算
    q_avg = q_score / len(q_task)
    worker_rate[worker] = q_avg
    
  # タスク数カウント辞書の準備
  # assign_dic[worker] = []
  # Q平均正解率 > th のワーカーのみ
  for th in threshold:
    top_worker_dic[th] = {}
  for th in threshold:
    for task in test_task:
      top_worker_dic[th][task] = []
      for worker in worker_rate:
        if worker_rate[worker] >= th:
          top_worker_dic[th][task].append(worker)
  
  return top_worker_dic

def top_worker_assignment(threshold, input_df, test_worker, q_task, test_task):
  # test_workerについてqualificationの正解率を計算する from input_df
  # ワーカーのqualifivation taskの正解率を格納する辞書
  n = 5
  worker_rate = {}
  # 上位N人のワーカ
  top_worker_dic = {}
  # test worker のtestタスクのqualification 正解率を調べる
  for worker in test_worker:
    q_score = 0
    # q_taskにおける平均正解率の計算
    for task in q_task:
      if input_df[worker][task] == 1:
        q_score += 1
    # 平均正解率の計算
    q_avg = q_score / len(q_task)
    worker_rate[worker] = q_avg

  # worker_rateを降順ソート
  worker_rate = dict(sorted(worker_rate.items(), key=lambda x: x[1], reverse=True))
 
  # 上位N人のワーカリスト
  top_worker_list = list(worker_rate.keys())
  top_worker_list = top_worker_list[:n]

  for th in threshold:
    top_worker_dic[th] = {}
    
  for th in threshold: 
    for task in test_task:
      top_worker_dic[th][task] = top_worker_list
     
  return top_worker_dic


# Average Accuracy Assignment
def AA_assignment(threshold, input_df, test_worker, q_task, test_task):

  AA_candidate_dic = {}
  # 上位N人のワーカ
  worker_rate = {}
  # test worker のtestタスクのqualification 正解率を調べる
  for worker in test_worker:
    q_score = 0
    # q_taskにおける平均正解率の計算
    for task in q_task:
      if input_df[worker][task] == 1:
        q_score += 1
    # 平均正解率の計算
    q_avg = q_score / len(q_task)
    worker_rate[worker] = q_avg

  # 上位N人のワーカ
  AA_top_workers_dict = dict(sorted(worker_rate.items(), key=lambda x: x[1], reverse=True)).keys()
 
  for th in threshold:
    AA_candidate_dic[th] = {}
  for th in threshold:
    for task in test_task:
      AA_candidate_dic[th][task] = []
      for worker in worker_rate:
        if worker_rate[worker] >= th:
          AA_candidate_dic[th][task].append(worker)
          
   

  return AA_candidate_dic, AA_top_workers_dict

# random assignment
def random_assignment(test_task, test_worker):
  assign_dic = {}
  for task in test_task:
    assign_dic[task] = random.choice(test_worker)
  return assign_dic


def Frequency_Distribution(data, class_width=None):
    data = np.asarray(data)
    if class_width is None:
        class_size = int(np.log2(data.size).round()) + 1
        class_width = round((data.max() - data.min()) / class_size)

    bins = np.arange(0, data.max()+class_width+1, class_width)
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

