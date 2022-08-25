# 各タスクの候補ワーカーのAの値平均を求める
# A: 各ワーカーの候補リスト中の登場回数
import copy
import numpy as np
def task_average(org_dic, c_dic):
  task_avg = {}
  for t in org_dic:
    s = 0
    for w in org_dic[t]:
      s += c_dic[w]
      # division by 0
      task_avg[t] = s / len(org_dic[t])
  task_avg = sorted(task_avg.items(), key=lambda x: x[1], reverse=False)
  # t_dic = dict(t_dic)
  return task_avg

#task_avg = task_avg(org_dic, c_dic)
# 候補リストが空かどうかチェック
def check_empty(dic):
  for i in dic.values():
    if len(i) == 0:
      return 1
# アルゴリズム


def flatten(assign_dic, worker_list, i):
  worker = worker_list[i]
  pre_worker =  worker_list[i-1]

  # if len(assign_dic[worker]) < len(assign_dic[pre_worker]):
  while len(assign_dic[worker]) < len(assign_dic[pre_worker]):
    task = assign_dic[pre_worker][0]
    assign_dic[worker].append(task)
    assign_dic[pre_worker].pop(0)
    if i >= 1:
      flatten(assign_dic, worker_list, i-1)
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




def assignment(worker_c, test_worker):
  org_dic = copy.deepcopy(worker_c)
  # workerリストの各ワーカーの候補リストの登場回数(A)の合計と平均を求める
  # c_dic: 各ワーカー: Aの合計の辞書
  c_dic = {}
  for w in test_worker:
    count = 0
    for l in org_dic.values():
      if w in l:
        count += 1
    c_dic[w] = count
  task_avg = task_average(org_dic, c_dic)
  # 割り当て処理用のタスク:ワーカー候補辞書
  mid_dic = copy.deepcopy(org_dic)
  # 割り当て処理
  assign_dic = {}
  # ワーカー登場回数(A)平均の小さいタスクから順に
  for task in task_avg:
    # id: タスクのid
    id = task[0]
    # print(dic[id])

    # candidate_dic = {worker: A}
    candidate_dic = {}

    # 現在のタスクの候補ワーカーリストを取得(アサインメントごとに更新)
    candidate_list = mid_dic[id]
    # 一巡はしていないが, そのタスクの候補リストが空で割り当てられないとき
    if len(candidate_list) == 0:
      # print('here => ', mid_dic)
      # 元のワーカー候補(origin_dicを代入)
      candidate_list = org_dic[id]
    # 候補ワーカーリストのサイズが1なら
    if len(candidate_list) == 1:
      # そのワーカーに割り当てる
      assign_worker = candidate_list[0]
    # このタスクの候補ワーカーについてforループ
    else:
      for w in candidate_list:
        candidate_dic[w] = c_dic[w]
    
      # 候補登場回数(A = c_dic[w])が最小のワーカーを見つけ, 割当先のワーカーとする
      assign_worker = min(candidate_dic.items(), key=lambda x: x[1])[0]
    assign_dic[id] = assign_worker
    # mid_dicからタスクを削除
    mid_dic.pop(id)
    
    # 候補リストを更新(タスクを割り当てられたワーカーは一旦候補から除く)
    for c_list in mid_dic.values():
      if assign_worker in c_list:
        c_list.remove(assign_worker)
    # print(mid_dic)
    #mid_dicが空ならリセット -- big sus
      empty_count = 0
      for dic in mid_dic.values():
        empty_count += len(dic)
    
    if empty_count == 0:
      # print('mid_dic is empty!')
      mid_dic = copy.deepcopy(org_dic)
      # print(mid_dic)
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
  for tw in test_worker:
    count_dic[tw] = 0
  for aw in assign_dic.values():
    count_dic[aw] += 1
  # 最大のvalueを調べる
  max_num = max(list(count_dic.values()))
  return max_num

