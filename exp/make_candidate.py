
import math
import girth
import pandas as pd
import numpy as np
import statistics
from girth import twopl_mml, onepl_mml, rasch_mml, ability_mle, rasch_jml
import random
from scipy.stats import norm
from irt_method import *
from simulation import * 

def DI_make_candidate_twopl(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task):
  # ワーカ候補の辞書
  worker_c_th = {}
  qualify_dic = {}
  
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
  # raschモデルでパラメータ推定
  params = run_girth_twopl(q_data, qualify_task, worker_list)
  item_param = params[0]
  user_param = params[1]

  DI_item_param = {}
  # カテゴリごとの推定難易度の辞書
  category_dic = {'Businness':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Economy':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 
                    'Technology&Science':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Health':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}}

  for i in qualify_task:
    # qualification taskの真のラベル
    category = label_df['true_label'][i]
    category_dic[category]['b'].append(item_param[i])

  for category in category_dic:
    category_dic[category]['mb'] = np.mean(category_dic[category]['b'])
  
  margin = calc_parameter_fit(qualify_task, worker_list, item_param, user_param, input_df)
  #print('DI残差:', margin)
  for th in threshold:
    
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # test_taskの推定カテゴリ
      est_label = label_df['estimate_label'][task]
      beta = category_dic[est_label]['mb']
      DI_item_param[task] = beta

      for worker in test_worker:
        # workerの正答確率prob
        theta = user_param[worker]
        prob = OnePLM(beta, theta)

        # workerの正解率がthresholdより大きければ
        if prob >= th:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)

    worker_c_th[th] = worker_c

  return worker_c_th, test_worker, qualify_task, test_task, DI_item_param, user_param


def DI_make_candidate_onepl(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task):
  # ワーカ候補の辞書
  worker_c_th = {}
  qualify_dic = {}
  
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
  # raschモデルでパラメータ推定
  params = run_girth_onepl(q_data, qualify_task, worker_list)
  item_param = params[0]
  user_param = params[1]

  DI_item_param = {}
  # カテゴリごとの推定難易度の辞書
  category_dic = {'Businness':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Economy':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 
                    'Technology&Science':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Health':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}}

  for i in qualify_task:
    # qualification taskの真のラベル
    category = label_df['true_label'][i]
    category_dic[category]['b'].append(item_param[i])

  for category in category_dic:
    category_dic[category]['mb'] = np.mean(category_dic[category]['b'])
  
  margin = calc_parameter_fit(qualify_task, worker_list, item_param, user_param, input_df)
  #print('DI残差:', margin)
  for th in threshold:
    margin = th/4
    
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # test_taskの推定カテゴリ
      est_label = label_df['estimate_label'][task]
      beta = category_dic[est_label]['mb']
      DI_item_param[task] = beta

      for worker in test_worker:
        # workerの正答確率prob
        theta = user_param[worker]
        prob = OnePLM(beta, theta)

        # workerの正解率がthresholdより大きければ
        if prob >= th:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)

    worker_c_th[th] = worker_c

  return worker_c_th, test_worker, qualify_task, test_task, DI_item_param, user_param

def DI_make_candidate_onepl_margin(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task):
  # ワーカ候補の辞書
  worker_c_th = {}
  qualify_dic = {}
  
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
  # raschモデルでパラメータ推定
  params = run_girth_onepl(q_data, qualify_task, worker_list)
  item_param = params[0]
  user_param = params[1]

  DI_item_param = {}
  # カテゴリごとの推定難易度の辞書
  category_dic = {'Businness':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Economy':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 
                    'Technology&Science':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Health':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}}

  for i in qualify_task:
    # qualification taskの真のラベル
    category = label_df['true_label'][i]
    category_dic[category]['b'].append(item_param[i])

  for category in category_dic:
    category_dic[category]['mb'] = np.mean(category_dic[category]['b'])
  
  margin = calc_parameter_fit(qualify_task, worker_list, item_param, user_param, input_df)
  #print('DI残差:', margin)
  for th in threshold:
    margin = th/4
    
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # test_taskの推定カテゴリ
      est_label = label_df['estimate_label'][task]
      beta = category_dic[est_label]['mb']
      DI_item_param[task] = beta

      for worker in test_worker:
        # workerの正答確率prob
        theta = user_param[worker]
        prob = OnePLM(beta, theta)

        # workerの正解率がthresholdより大きければ
        if prob >= th+margin:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)

    worker_c_th[th] = worker_c

  return worker_c_th, test_worker, qualify_task, test_task, DI_item_param, user_param

def PI_make_candidate(threshold, full_item_param, full_user_param, test_worker, test_task):
  worker_c_th = {}

  for th in threshold:
    margin = 0
  
    worker_c = {}
    for task in test_task:
      #if task_assignable_check(th, full_item_param, full_user_param, test_worker, task) == True:
      # Aタスク
      worker_c[task] = []
      beta = full_item_param[task]
      for worker in test_worker:
        # workerの正答確率prob
        theta = full_user_param[worker]
        prob = OnePLM(beta, theta)

        # workerの正解率がthresholdより大きければ
        if prob >= th+margin:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)
    worker_c_th[th] = worker_c

  return worker_c_th, full_item_param, full_user_param

def PI_make_candidate_margin(threshold, full_item_param, full_user_param, test_worker, test_task):
  worker_c_th = {}

  for th in threshold:
    margin = th/4
    
    worker_c = {}
    for task in test_task:
      #if task_assignable_check(th, full_item_param, full_user_param, test_worker, task) == True:
      # Aタスク
      worker_c[task] = []
      beta = full_item_param[task]
      for worker in test_worker:
        # workerの正答確率prob
        theta = full_user_param[worker]
        prob = OnePLM(beta, theta)

        # workerの正解率がthresholdより大きければ
        if prob >= th+margin:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)

    worker_c_th[th] = worker_c

  return worker_c_th

def make_candidate_PI_noise(threshold, input_df, full_item_param, full_user_param, test_worker, test_task):
  worker_c_th = {}
  top_result = {}
  margin = 0
  # print(item_param)
  # 難易度の最小値, 最大値

  for th in threshold:
    # margin = th / 5
    margin = th/8
    worker_c = {}
    for task in test_task:
      if task_assignable_check(th+margin, full_item_param, full_user_param, test_worker, task) == True:
      
        # Aタスク
        worker_c[task] = []
        actual_b = full_item_param[task]
        beta = ai_model(actual_b, dist=0.05)
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


def top_make_cabdidate(threshold, input_df, test_worker, q_task, test_task):
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
def AA_make_candidate(threshold, input_df, test_worker, q_task, test_task):

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
  top_workers_dict = dict(sorted(worker_rate.items(), key=lambda x: x[1], reverse=True))
 
  for th in threshold:
    AA_candidate_dic[th] = {}
  for th in threshold:
    for task in test_task:
      AA_candidate_dic[th][task] = []
      for worker in worker_rate:
        if worker_rate[worker] >= th:
          AA_candidate_dic[th][task].append(worker)
  
  return AA_candidate_dic, top_workers_dict, worker_rate

# random assignment
def random_assignment(test_task, test_worker):
  assign_dic = {}
  for task in test_task:
    assign_dic[task] = random.choice(test_worker)
  return assign_dic

def extract_sub_worker_irt(test_worker, test_task, item_param, user_param):
  sub_worker = {}
  sorted_worker_dict = dict(sorted(user_param.items(), key=lambda x: x[1], reverse=True))
  sorted_worker_list = list(sorted_worker_dict.keys())

  sorted_test_worker = []
  for worker in sorted_worker_list:
    if worker in test_worker:
      sorted_test_worker.append(worker)

  for task in test_task:
    sub_worker[task] = []
    beta = item_param[task]
    for worker in sorted_test_worker:
      theta = user_param[worker]
      prob = OnePLM(beta, theta)
      if prob > 0.5:
          # ワーカーを候補リストに代入
          sub_worker[task].append(worker)
  
  return sub_worker
  
def extract_sub_worker_AA(worker_rate):
    sub_worker = []
    sorted_workers_dict = dict(sorted(worker_rate.items(), key=lambda x: x[1], reverse=True))
    sorted_worker_list = list(sorted_workers_dict.keys())
    for worker in sorted_worker_list:
      if worker_rate[worker] > 0.5:
        sub_worker.append(worker)
      
      return sub_worker