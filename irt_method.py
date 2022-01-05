import math
import girth
import pandas as pd
import numpy as np
from girth import twopl_mml, onepl_mml, ability_mle
import random


def OnePLM(beta, theta, d=1.7, c=0.0):
  p = 1 / (1+np.exp(-d*(theta-beta)))
  # p = c + (1.0 - c) / (1 + np.exp(-(1 * theta + beta)))
  return p
def TwoPLM(a, b, theta, d=1.7):
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
    print(len(user_param))
    return item_param, user_param

# qualification task, test taskに分けてシミュレーション
# IRT: girthを使用
# 
def make_candidate(threshold, input_df, label_df, worker_list, task_list):
  worker_c_th = {}
  # 承認タスクとテストタスクを分離
  random.shuffle(task_list)
  qualify_task = task_list[:60]
  test_task = task_list[60:]

  # t_worker = worker_list

  qualify_dic = {}
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])
  q_data = np.array(list(qualify_dic.values()))
  # q_data = input_df.values
  
  params = run_girth_onepl(q_data, qualify_task, worker_list)
  item_param = params[0]
  user_param = params[1]

  worker_list = elim_spam(item_param, user_param, input_df, worker_list)
  t_worker = random.sample(worker_list, 20)

  category_dic = {'Businness':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Economy':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 
                    'Technology&Science':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Health':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}}
  for i in qualify_task:
    category = label_df['estimate_label'][i]
    # category_dic[category]['a'].append(item_param[i]['alpha'])
    category_dic[category]['b'].append(item_param[i])
    category_dic[category]['num'] += 1

  for c in category_dic:
    category_dic[c]['ma'] = np.sum(category_dic[c]['a']) / category_dic[c]['num']
    category_dic[c]['mb'] = np.sum(category_dic[c]['b']) / category_dic[c]['num']

  # 各テストタスクについてワーカー候補を作成する
  # output: worker_c = {task: []}
  # すべてのスレッショルドについてワーカー候補作成
  for th in threshold:
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # test_taskのカテゴリ推定
      est_label = label_df['estimate_label'][task]
      # user_paramのワーカー能力がcategory_dicのタスク難易度より大きければ候補に入れる.
      # a = item_param[task]['alpha']
      beta = category_dic[est_label]['mb']

      for worker in t_worker:
        # workerの正答確率prob
        theta = user_param[worker]
        # prob = irt_fnc(theta, b, a)
        prob = OnePLM(beta, theta)
        # prob = TwoPLM(a, b, theta, d=1.7)

        # workerの正解率がthresholdより大きければ
        if prob >= th:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)
  
    worker_c_th[th] = worker_c

  return worker_c_th, t_worker, test_task

def make_candidate_all(threshold, input_df, label_df, worker_list, task_list):
  worker_c_th = {}
  # 承認タスクとテストタスクを分離
  qualify_task = task_list
  random.shuffle(task_list)
  # qualify_task = task_list[:100]
  test_task = task_list[60:]

  qualify_dic = {}
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
 
  # t_worker = random.sample(worker_list, 20)
  # t_worker = worker_list
  # q_data = np.array(list(qualify_dic.values()))


  params = run_girth_onepl(q_data, qualify_task, worker_list)
  item_param = params[0]
  user_param = params[1]
  print(item_param)

  worker_list = elim_spam(item_param, user_param, input_df, worker_list)
  t_worker = random.sample(worker_list, 20)


  # 各テストタスクについてワーカー候補を作成する
  # output: worker_c = {task: []}
  # すべてのスレッショルドについてワーカー候補作成
  for th in threshold:
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # user_paramのワーカー能力がcategory_dicのタスク難易度より大きければ候補に入れる.
      # alpha = item_param[task]['a']
      beta = item_param[task]
    
      for worker in t_worker:
        # workerの正答確率prob
        
        theta = user_param[worker]
        # prob = irt_fnc(theta, b, a)
        prob = OnePLM(beta, theta)
        # print(prob)
        # prob = TwoPLM(alpha, beta, theta, d=1.7)

        # workerの正解率がthresholdより大きければ
        if prob >= th:
          # ワーカーを候補リストに代入
          worker_c[task].append(worker)
  
    worker_c_th[th] = worker_c

  return worker_c_th, t_worker, test_task

def estimate_candidate(threshold, input_df, label_df, worker_list, task_list):
  worker_c_th = {}
  # 承認タスクとテストタスクを分離
  
  random.shuffle(task_list)
  qualify_task = task_list[:60]
  test_task = task_list[60:]
  t_worker = random.sample(worker_list, 20)
 
  # IRT
  data = input_df.values
  q_data = data[:60]
  t_data = data[60:]
  params = run_girth(q_data, qualify_task, worker_list)
  item_param = params[0]
  user_param = params[1]
  category_dic = {'Businness':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Economy':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 
                    'Technology&Science':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}, 'Health':{'a': [], 'b': [], 'num':0, 'ma': 0, 'mb': 0}}
  for i in qualify_task:
    category = label_df['estimate_label'][i]
    # category_dic[category]['a'].append(item_param[i]['alpha'])
    category_dic[category]['b'].append(item_param[i])
    category_dic[category]['num'] += 1

  for c in category_dic:
    category_dic[c]['ma'] = np.sum(category_dic[c]['a']) / category_dic[c]['num']
    category_dic[c]['mb'] = np.sum(category_dic[c]['b']) / category_dic[c]['num']
  # 各テストタスクについてワーカー候補を作成する
  # output: worker_c = {task: []}
  # すべてのスレッショルドについてワーカー候補作成
  for th in threshold:
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # test_taskのカテゴリ推定
      est_label = label_df['estimate_label'][task]
      # user_paramのワーカー能力がcategory_dicのタスク難易度より大きければ候補に入れる.
      for worker in t_worker:
        # workerの正答確率prob
        a = category_dic[est_label]['ma']
        b = category_dic[est_label]['mb']
        theta = user_param[worker]
        prob = OnePLM(b, theta)
        # th = th - 0.5
        print(theta - b)
        # workerの正解率がthresholdより大きければ
        if prob >= th:
          worker_c[task].append(worker)
    worker_c_th[th] = worker_c

  return worker_c_th, t_worker, test_task

# 推定難易度とIRTによる難易度の比較
# すべてのタスクでIRT / 割り当ては実行しない
def compare_difficulty(input_df, label_df, worker_list, task_list):
  all_params = run_girth_onepl(input_df.values, task_list, worker_list)
  item_param_all = all_params[0]
  # 承認タスクとテストタスクを分離
  
  random.shuffle(task_list)
  qualify_task = task_list[:60]
  test_task = task_list[60:]

  qualify_dic = {}
  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
  t_worker = random.sample(worker_list, 20)

  devided_params = run_girth_onepl(q_data, qualify_task, worker_list)
  item_param = devided_params[0]
  user_param = devided_params[1]
  

  # カテゴリごとに難易度推定
  category_dic = {'Businness':{'b': {}, 'num':0, 'mb': 0}, 'Economy':{'b': {}, 'num':0, 'mb': 0}, 
                    'Technology&Science':{'b': {}, 'num':0, 'mb': 0}, 'Health':{'b': {}, 'num':0, 'mb': 0}}
  for i in qualify_task:
    category = label_df['estimate_label'][i]
    # category_dic[category]['a'].append(item_param[i]['alpha'])
    category_dic[category]['b'][i] = item_param[i]
    category_dic[category]['num'] += 1

  for c in category_dic:
    # category_dic[c]['ma'] = np.sum(category_dic[c]['a']) / category_dic[c]['num']
    # print(list(category_dic[c]['b'].values()))
    category_dic[c]['mb'] = np.sum(list(category_dic[c]['b'].values())) / category_dic[c]['num']

  print(category_dic)
  return category_dic, item_param_all, test_task

# spamワーカー除去
def elim_spam(item_param, user_param, input_df, worker_list):
  # IRTの困難度と乖離した回答パタンのワーカーを除外する
  # item_paramを難易度bでソート
  sorted_item_param = sorted(item_param.items(), key=lambda x: x[1], reverse=False)
  sorted_user_param = sorted(user_param.items(), key=lambda x: x[1], reverse=True)
  print(sorted_item_param)

  # すべてのタスクについてIRTで難易度推定
  # 難易度下位N件
  # sorted_task_list = list(sorted_item_param)
  i_n = 30
  i_m = len(sorted_item_param) - i_n
  easy_task = sorted_item_param[:i_n]
  # 難易度上位M件
  difficult_task = sorted_item_param[i_m:]

  u_n = 10
  u_m = len(sorted_user_param) - u_n
  high_user = sorted_user_param[:u_n]
  low_user = sorted_user_param[:u_m]

  for user_tuple in high_user:
    user = user_tuple[0]
    score_easy = 0
    score_difficult = 0
    for tuple_e in easy_task:
      task = tuple_e[0]
      if input_df[user][task] == 1:
        score_easy += 1
    for tuple_d in difficult_task:
      task = tuple_d[0]
      if input_df[user][task] == 1:
        score_difficult += 1
    
    if score_easy / len(easy_task) < 0.5:
      worker_list.remove(user)
  # 簡単なタスクの正解率 < 難しいタスクの正解率 のワーカー(from 下位X%)
  for user_tuple in low_user:
    user = user_tuple[0]
    score_easy = 0
    score_difficult = 0
    for tuple_e in easy_task:
      task = tuple_e[0]
      if input_df[user][task] == 1:
        score_easy += 1
    for tuple_d in difficult_task:
      task = tuple_d[0]
      if input_df[user][task] == 1:
        score_difficult += 1
    
    #if score_easy / len(easy_task) < score_difficult / len(difficult_task):
      #worker_list.remove(user)

  return worker_list

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

