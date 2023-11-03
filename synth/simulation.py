import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from irt_method import *

def result_plot_acc_var(threshold, result_dic, ay, bbox):
    
    # 推移をプロット
    plt.rcParams["font.size"] = 24
    fig = plt.figure() #親グラフと子グラフを同時に定義
    ax = fig.add_subplot()
    ax.set_xlabel('threshold')
    ax.set_ylabel(ay)
    x = np.array(threshold)
    
    DI = np.array(result_dic['DI'])
    top = np.array(result_dic['top'])
    AA = np.array(result_dic['AA'])
    random = np.array(result_dic['random'])
    PI = np.array(result_dic['PI'])

    DI_std = np.array(result_dic['DI_std'])
    top_std = np.array(result_dic['top_std'])
    random_std = np.array(result_dic['random_std'])
    PI_std = np.array(result_dic['PI_std'])
    AA_std = np.array(result_dic['AA_std'])
    ax.plot(x, DI, color='red', marker='s', label='IRT(DI)')
    ax.plot(x, top, color='blue', marker='s', label='TOP')
    ax.plot(x, AA, color='cyan',  marker='s', label='AA')
    ax.plot(x, random, color='green', marker='s', label='RANDOM')
    ax.plot(x, PI, color='purple', marker='s', label='IRT(PI)')
    if ay == 'accuracy':
        ax.plot(x, x, color='orange', linestyle="dashed")

    a = 0.05
    plt.fill_between(x, DI - DI_std, DI + DI_std, facecolor='r', alpha=a)
    plt.fill_between(x, top - top_std, top + top_std, facecolor='b', alpha=a)
    plt.fill_between(x, AA - AA_std, AA + AA_std, facecolor='cyan', alpha=a)
    plt.fill_between(x, random - random_std, random + random_std, facecolor='g', alpha=a)
    plt.fill_between(x, PI - PI_std, PI + PI_std, facecolor='purple', alpha=a)
    fig.legend(bbox_to_anchor=bbox, loc='upper left')
    return plt

def result_plot_tradeoff(result_tp_dic, result_acc_dic):
  # トレードオフのグラフ
  DI_tp = np.array(result_tp_dic['DI'])
  top_tp = np.array(result_tp_dic['top'])
  AA_tp = np.array(result_tp_dic['AA'])
  random_tp = np.array(result_tp_dic['random'])
  PI_tp = np.array(result_tp_dic['PI'])

  DI_acc = np.array(result_acc_dic['DI'])
  top_acc = np.array(result_acc_dic['top'])
  AA_acc = np.array(result_acc_dic['AA'])
  random_acc = np.array(result_acc_dic['random'])
  PI_acc = np.array(result_acc_dic['PI'])

  DI_trade = tp_acc_plot(DI_tp, DI_acc)
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
  ax.set_xlim(0, 15)

  bbox=(0.2750, 0.400)
  ax.plot(DI_trade[0], DI_trade[1], color='red', marker='s', label='IRT(DI)')
  ax.plot(AA_trade[0], AA_trade[1], color='cyan', marker='s', label='AA')
  ax.plot(top_trade[0], top_trade[1], color='blue', marker='s', label='TOP')
  ax.plot(random_trade[0], random_trade[1], color='green', marker='s', label='RANDOM')
  ax.plot(PI_trade[0], PI_trade[1], color='purple', marker='s', label='IRT(PI)')
  fig.legend(bbox_to_anchor=bbox, loc='upper left')
  #plt.show()
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

# 実際の能力 > 実際の難易度だった割当ての度数分布thresholdごとに
# 引数: 割り当ての辞書，実際のワーカ能力，実際のタスク難易度
def welldone_count(threshold, assign_dic, user_param, item_param):
   
    count = 0
    print('length' + str(len(assign_dic)))
    for task in assign_dic:
        worker = assign_dic[task]
        b = item_param[task]
        theta = user_param[worker]

        # 能力 > 難易度ならdist[i] ++ 1
        if OnePLM(b, theta) >= threshold:
            count += 1
    return count

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
