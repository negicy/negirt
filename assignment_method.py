# 各タスクの候補ワーカーのAの値平均を求める
# A: 各ワーカーの候補リスト中の登場回数
import copy
import numpy as np


def task_average(org_dic, score_dic):
  task_avg = {}
  for t in org_dic:
    sum = 0
    for w in org_dic[t]:
      sum += score_dic[w]
      # division by 0
      task_avg[t] = sum / len(org_dic[t])
  task_avg = sorted(task_avg.items(), key=lambda x: x[1], reverse=False)
  # t_dic = dict(t_dic)
  # ソートされた辞書を返す
  return task_avg

#task_avg = task_avg(org_dic, c_dic)
# 候補リストが空かどうかチェック
def check_empty(dic):
  for i in dic.values():
    if len(i) == 0:
      return 1
# アルゴリズム

def assignment(worker_c, test_worker):
  org_dic = copy.deepcopy(worker_c)
  # workerリストの各ワーカーの候補リストの登場回数(A)の合計と平均を求める
  # c_dic: 各ワーカー: Aの合計の辞書
  score_dic = {}
  for worker in test_worker:
    count = 0
    for candidate_list in org_dic.values():
      if worker in candidate_list:
        count += 1
    score_dic[worker] = count
  task_avg = task_average(org_dic, score_dic)
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
      print('here => ', mid_dic)
      # 元のワーカー候補(origin_dicを代入)
      candidate_list = org_dic[id]
    # 候補ワーカーリストのサイズが1なら
    if len(candidate_list) == 1:
      # そのワーカーに割り当てる
      assign_worker = candidate_list[0]
    # このタスクの候補ワーカーについてforループ
    else:
      for w in candidate_list:
        candidate_dic[w] = score_dic[w]
      print(candidate_dic)
      # 候補登場回数(A = c_dic[w])が最小のワーカーを見つけ, 割当先のワーカーとする
      assign_worker = min(candidate_dic.items(), key=lambda x: x[1])[0]

    assign_dic[id] = assign_worker
    # mid_dicから割り当ての完了したタスクを削除
    mid_dic.pop(id)
    
    # 候補リストを更新(タスクを割り当てられたワーカーは一旦候補から除く)
    for c_list in mid_dic.values():
      if assign_worker in c_list:
        c_list.remove(assign_worker)
    # print(mid_dic)
    # mid_dicが空ならリセット
      empty_count = 0
      for dic in mid_dic.values():
        empty_count += len(dic)
    
    if empty_count == 0:
      print('mid_dic is empty!')
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
  # print(score/task_num)
  acc = score/task_num
  return acc

# 各テストタスクの正解率平均求める
def simple_vote(mul_assign_dic, input_df):
  overall_score = 0
  task_num = 0
  for task in mul_assign_dic:
    score = 0
    worker_list = mul_assign_dic[task]
    task_num += 1
    #print(task, worker)
    for worker in worker_list:
      if input_df[worker][task] == 1:
        score += 1
    if score >= 2:
      overall_score += 1
    
  # print(score/task_num)
  acc = overall_score/task_num
  return acc

def weighted_majority_vote(mul_assign_dic, input_df, user_param):
  task_num = 0
  overall_score = 0
  for task in mul_assign_dic:
    worker_list = mul_assign_dic[task]
    task_num += 1
    vote_correct = 0
    vote_wrong = 0
    for worker in worker_list:
      if input_df[worker][task] == 1:
        vote_correct += user_param[worker]
      elif input_df[worker][task] == 0:
        vote_wrong += user_param[worker]
    if vote_correct > vote_wrong:
      overall_score += 1
  acc = overall_score / task_num
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
  print(count_dic)
  print('========')
  v = np.var(count_list)
  return v

