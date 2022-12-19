import math
import girth
import pandas as pd
import numpy as np
import statistics
from girth import twopl_mml, onepl_mml, rasch_mml, ability_mle
import random
from scipy.stats import norm

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
def run_girth_rasch(data, task_list, worker_list):
    estimates = rasch_mml(data, 1)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']

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


    return item_param, user_param

def run_girth_onepl(data, task_list, worker_list):
    estimates = onepl_mml(data)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']


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

    return item_param, user_param

# input: 01の2D-ndarray, 対象タスクのリスト, 対象ワーカーのリスト
def run_girth_twopl(data, task_list, worker_list):
    estimates = twopl_mml(data)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']

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

    return item_param, user_param

# qualification task, test taskに分けてシミュレーション
# IRT: girthを使用
# 


# ワーカとタスクを分離
def devide_sample(task_list, worker_list, label_df):
  output = {}
  n = 40
  #qualify_task = sample_category(task_list, n, label_df)
  task_list_shuffle = random.sample(task_list, len(task_list))
  qualify_task = task_list_shuffle[:n]
  test_task = list(set(task_list) - set(qualify_task))

  #print(len(qualify_task))
  output['qualify_task'] = qualify_task
  output['test_task'] = test_task
  output['test_worker'] = random.sample(worker_list, 20)

  return output

def task_assignable_check(th, item_param, user_param, test_worker, task):
  b = item_param[task]
  for worker in test_worker:
    theta = user_param[worker]
    prob = OnePLM(b, theta)
    if prob >= th:
      return True
  
  return False

def ai_model(actual_b, dist):
    b_norm = norm.rvs(loc=actual_b, scale=dist, size=100)
    # print(b_norm)
    ai_accuracy = list(b_norm).count(actual_b) / len(b_norm)
    # print(ai_accuracy)
    return random.choice(b_norm)
  
def make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task):
  # ワーカ候補の辞書
  worker_c_th = {}
  qualify_dic = {}

  for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

  q_data = np.array(list(qualify_dic.values()))
  # raschモデルでパラメータ推定
  params = run_girth_rasch(q_data, qualify_task, worker_list)
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
  mb_list = []

  for category in category_dic:
    category_dic[category]['mb'] = np.mean(category_dic[category]['b'])
    for num in range(0, category_dic[category]['num']):
      mb_list.append(category_dic[category]['mb'])

  for th in threshold:
    candidate_count = 0
    worker_c = {}
    for task in test_task:
      worker_c[task] = []
      # test_taskのカテゴリ推定
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


# 割当て候補のいないタスクを無くす
def sort_test_worker(test_worker, user_param, N):
  test_worker_param = {}
  for worker in test_worker:
    test_worker_param[worker] = user_param[worker]

  sorted_user_param = dict(sorted(test_worker_param.items(), key=lambda x: x[1], reverse=True))
  top_workers = list(sorted_user_param.keys())[:N]
  return top_workers

def make_candidate_all(threshold, input_df, full_item_param, full_user_param, test_worker, test_task):
  worker_c_th = {}

  for th in threshold:
    # margin = th / 5
    if th < 0.525:
      margin = 0.32
    
    elif th <= 0.55:
      #margin = th/1.60
      # margin = 2.25
      #margin = th/2.55
      margin = 0.30
    elif th <= 0.60:
      margin = 0.280

    elif th <= 0.63:
      mergin = th/2.5
    elif th <= 0.67:
      margin = 0.24
    else:
      margin = 0.18
    # margin = th/4.0
    #margin = th/3.9
    
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

          # workerの正解率がthresholdより大きければ
          if prob >= th + margin:
            # ワーカーを候補リストに代入
            worker_c[task].append(worker)
    worker_c_th[th] = worker_c
   
    

  return worker_c_th, full_item_param, full_user_param


def make_candidate_PI_noise(threshold, input_df, full_item_param, full_user_param, test_worker, test_task):
  worker_c_th = {}
  top_result = {}
  margin = 0
  # print(item_param)
  # 難易度の最小値, 最大値
  theta_range = []
  #theta_range.append(np.min(list(full_item_param.values())))
  #theta_range.append(np.max(list(full_item_param.values())))  

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
          if prob >= th - margin:
            # ワーカーを候補リストに代入
            worker_c[task].append(worker)
         
    worker_c_th[th] = worker_c



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
  top_workers_dict = dict(sorted(worker_rate.items(), key=lambda x: x[1], reverse=True))
  # rop_workers_list = list(dict(sorted(worker_rate.items(), key=lambda x: x[1], reverse=True)).keys())
 
  for th in threshold:
    AA_candidate_dic[th] = {}
  for th in threshold:
    for task in test_task:
      AA_candidate_dic[th][task] = []
      for worker in worker_rate:
        if worker_rate[worker] >= th:
          AA_candidate_dic[th][task].append(worker)
      # 割当てがないタスクに上位ワーカを割り当てる．
      #if len(AA_candidate_dic[th][task]) == 0:
        #AA_candidate_dic[th][task] = top_workers[:3]

  return AA_candidate_dic, top_workers_dict

# random assignment
def random_assignment(test_task, test_worker):
  assign_dic = {}
  for task in test_task:
    assign_dic[task] = random.choice(test_worker)
  return assign_dic

# ワーカ人数の比較用ヒストグラム
def just_candidate(threshold, label_df, worker_list, task_list):
  worker_c_th = {}
  # 承認タスクとテストタスクを分離
  import random
  random.shuffle(task_list)
  qualify_task = task_list[:60]
  test_task = task_list[60:]
  t_worker = random.sample(worker_list, 20)
 
  # top-worker-approach による仕事のあるワーカー一覧: 各スレッショルドごと
  top_worker_th = {}
  # top worker assignment
  for th in threshold:
    top_results = entire_top_workers(test_task, t_worker, qualify_task, th)
    # top_worker_quality = top_results[0]
    # top_worker_variance = top_results[1]
    top_worker_th[th] = top_results
  
  # IRT
  params = irt_devide(worker_list, qualify_task)
  item_param = params[0]
  user_param = params[1]

  category_dic = {'Businness':{'b': [], 'num':0, 'm': 0}, 'Economy':{'b': [], 'num':0, 'm': 0}, 
                  'Technology&Science':{'b': [], 'num':0, 'm': 0}, 'Health':{'b': [], 'num':0, 'm': 0}}
  for i in qualify_task:
    category = label_df['true_label'][i]
    category_dic[category]['b'].append(item_param[i]['beta'])
    category_dic[category]['num'] += 1

  for c in category_dic:
    category_dic[c]['m'] = np.sum(category_dic[c]['b']) / category_dic[c]['num']
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
       
        b = category_dic[est_label]['m']
        theta = user_param[worker]
        # prob = OnePLM(b, theta, d=1.7)
        # workerの正解率がthresholdより大きければ
        prob = OnePLM(b, theta) 
        if prob >= th:
          worker_c[task].append(worker)
    worker_c_th[th] = worker_c

  return worker_c_th, top_worker_th

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

