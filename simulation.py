import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from irt_method import *

# ワーカとタスクを分離
def devide_sample(task_list, worker_list, label_df):
  output = {}
  n = 40

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

# 割当て候補のいないタスクを無くす
def sort_test_worker(test_worker, user_param, N):
  test_worker_param = {}
  for worker in test_worker:
    test_worker_param[worker] = user_param[worker]

  sorted_user_param = dict(sorted(test_worker_param.items(), key=lambda x: x[1], reverse=True))
  top_workers = list(sorted_user_param.keys())[:N]
  return top_workers
  


def combine_iteration(threshold, iteration_time, acc_allth, var_allth, tp_allth):
  acc = [0] * len(threshold)
  var = [0] * len(threshold)
  tp = [0] * len(threshold)
  acc_std = []
  var_std = []
  for th in range(0, len(threshold)):
    
    acc_sum = 0
    var_sum = 0
    tp_sum = 0
    # thresholdごとのacc, varのリスト, 標準偏差の計算に使う
    list_acc_th = []
    list_var_th = []
    for i in range(0, iteration_time): 
        acc_sum += acc_allth[i][th]
        list_acc_th.append(acc_allth[i][th])
        var_sum += var_allth[i][th]
        list_var_th.append(var_allth[i][th])
        tp_sum += tp_allth[i][th]
        
    acc[th] = acc_sum / iteration_time
    var[th] = var_sum / iteration_time
    tp[th] = tp_sum / iteration_time

    # 標準偏差を計算
    acc_std_th = np.std(list_acc_th)
    var_std_th = np.std(list_var_th)
    acc_std.append(acc_std_th)
    var_std.append(var_std_th)
    if th == 0:
      acc_head = list_acc_th
    if th == len(threshold)-1:
      acc_tail = list_acc_th
  print(acc)

    
  return acc, var, tp, acc_std, var_std,  acc_head, acc_tail



def result_plot_1(threshold, result_dic, ay, bbox):
    
    # 推移をプロット
    plt.rcParams["font.size"] = 24
    fig = plt.figure() #親グラフと子グラフを同時に定義
    ax = fig.add_subplot()
    ax.set_xlabel('threshold')
    ax.set_ylabel(ay)
    x = np.array(threshold)
    
    ours = np.array(result_dic['ours'])
    top = np.array(result_dic['top'])
    AA = np.array(result_dic['AA'])
    random = np.array(result_dic['random'])
    PI = np.array(result_dic['PI'])

    ours_std = np.array(result_dic['ours_std'])
    top_std = np.array(result_dic['top_std'])
    random_std = np.array(result_dic['random_std'])
    PI_std = np.array(result_dic['PI_std'])
    AA_std = np.array(result_dic['AA_std'])
    ax.plot(x, ours, color='red', label='IRT(DI)')
    ax.plot(x, top, color='blue', label='TOP')
    ax.plot(x, AA, color='cyan', label='AA')
    ax.plot(x, random, color='green', label='RANDOM')
    ax.plot(x, PI, color='purple', label='IRT(PI)')
    if ay == 'accuracy':
        ax.plot(x, x, color='orange', linestyle="dashed")

    a = 0.05
    plt.fill_between(x, ours - ours_std, ours + ours_std, facecolor='r', alpha=a)
    plt.fill_between(x, top - top_std, top + top_std, facecolor='b', alpha=a)
    plt.fill_between(x, AA - AA_std, AA + AA_std, facecolor='cyan', alpha=a)
    plt.fill_between(x, random - random_std, random + random_std, facecolor='g', alpha=a)
    plt.fill_between(x, PI - PI_std, PI + PI_std, facecolor='purple', alpha=a)
    fig.legend(bbox_to_anchor=bbox, loc='upper left')
    return plt


def result_plot_2(threshold, result_dic, ay, bbox):
    
    # 推移をプロット
    plt.rcParams["font.size"] = 22
    fig = plt.figure() #親グラフと子グラフを同時に定義
    ax = fig.add_subplot()
    ax.set_xlabel('threshold')
    ax.set_ylabel(ay)
    x = np.array(threshold)
    
    ours = np.array(result_dic['ours'])
    PI = np.array(result_dic['PI'])

    ours_std = np.array(result_dic['ours_std'])
    PI_std = np.array(result_dic['PI_std'])
    
    ax.plot(x, ours, color='red', label='ours')
    ax.plot(x, PI, color='purple', label='IRT')
    if ay == 'accuracy':
        ax.plot(x, x, color='orange', linestyle="dashed")
    print(x)
    a = 0.05
    plt.fill_between(x, ours - ours_std, ours + ours_std, facecolor='r', alpha=a)
    plt.fill_between(x, PI - PI_std, PI + PI_std, facecolor='purple', alpha=a)
    fig.legend(bbox_to_anchor=bbox, loc='upper left')
    return plt


def var_acc_plot(var, acc):
    # ソート
    c = zip(var, acc)
    c = sorted(c)
    var, acc = zip(*c)
    var = list(var)
    acc = list(acc)
    # 推移をプロット
    '''
    plt.rcParams["font.size"] = 18
    fig = plt.figure() #親グラフと子グラフを同時に定義
    ax = fig.add_subplot()
    ax.set_xlabel('variance')
    ax.set_ylabel('accuracy')
    ax.plot(var, acc, color='red', label='ours')
    '''
    
    
    return var, acc

def tp_acc_plot(tp, acc):
    # ソート
    c = zip(tp, acc)
    c = sorted(c)
    tp, acc = zip(*c)
    tp = list(tp)
    acc_list = list(acc)
    # accを逆数にする
    acc_reverse = []
    for acc in acc_list:
       
        acc_reverse.append(acc)

   
    # 推移をプロット
    '''
    
    plt.rcParams["font.size"] = 18
    fig = plt.figure() #親グラフと子グラフを同時に定義
    ax = fig.add_subplot()
    ax.set_xlabel('TP')
    ax.set_ylabel('accuracy')
    ax.plot(tp, acc, color='red', label='ours')
    '''
    
    return tp, acc_reverse

# 実際の能力 > 実際の難易度だった割当ての度数分布thresholdごとに
# 引数: 割り当ての辞書，実際のワーカ能力，実際のタスク難易度
def welldone_count(threshold, assign_dic, user_param, item_param):
   
    count = 0
    # print('length' + str(len(assign_dic)))
    for task in assign_dic:
        worker = assign_dic[task]
        b = item_param[task]
        theta = user_param[worker]

        # 能力 > 難易度ならdist[i] ++ 1
        if OnePLM(b, theta) >= threshold:
            count += 1
    return count

def has_duplicates(seq):
    return len(seq) != len(set(seq))

