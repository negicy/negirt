# データ読み込み

# 考察: タスク割り当ての実際のパラメータプロット図
plt.rcParams["font.size"] = 18
threshold = [0.75]
sample = devide_sample(task_list, worker_list)
qualify_task = sample['qualify_task']
test_task = sample['test_task']
test_worker = sample['test_worker']

ours_output = make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
# ワーカ候補のリスト
ours_candidate = ours_output[0]

all_output = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, test_task)
irt_candidate = all_output[0]
item_param = all_output[1]
user_param = all_output[2]
for candidate_dic in ours_candidate.values():
  ours_assign_dic = assignment(candidate_dic, test_worker)

for candidate_dic in irt_candidate.values():
  # print(candidate_dic)
  all_assign_dic = assignment(candidate_dic, test_worker)

print(item_param)
ours_item_list = []
ours_user_list = []

all_item_list = []
all_user_list = []
# タスクとワーカの実際のパラメータを格納
for task in all_assign_dic:
  ours_item_list.append(item_param[task])
  worker = ours_assign_dic[task]
  ours_user_list.append(user_param[worker])

for task in all_assign_dic:
  
  all_item_list.append(item_param[task])
  worker = all_assign_dic[task]
  all_user_list.append(user_param[worker])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
print(len(all_item_list))
print(len(ours_item_list))
#ax.scatter(ours_item_list, ours_user_list, marker="^", label="ours")
ax.scatter(all_item_list, all_user_list, label="Assignment")
plt.legend()
ax.set_xlabel('task difficulty')
ax.set_ylabel('worker skill')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

#ax.scatter(ours_item_list, ours_user_list, marker="^", label="ours")
ax.scatter(ours_item_list, ours_user_list, label="Assignment")
plt.legend()
ax.set_xlabel('task difficulty')
ax.set_ylabel('worker skill')

plt.show()


# ワーカ候補作成ヒストグラム
threshold = list([i / 100 for i in range(60, 81, 10)])


# ワーカー候補だけ作成する
# ワーカーリスト作成~割り当て　実行
ours_all_iter = []
baseline_all_iter = []
iteration_time = 5
for iteration in range(0, iteration_time):
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker'] 

  output =  make_candidate(threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task)
  ours_candidate = output[0]

  baseline_candidate = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, qualify_task, test_task)[0]
  # baseline_candidate = entire_top_workers(threshold, input_df, test_worker, qualify_task, test_task)
  # print(results)

  # top-worker 
  # baseline_candidate = entire_top_workers

  ours_all_iter.append(ours_candidate)
  baseline_all_iter.append(baseline_candidate)
  # print(ours_all_iter)
  # ワーカ人数をカウントする辞書
baseline_num_dic = {}
ours_num_dic = {}

for th in threshold:
  baseline_num_dic[th] = 0
  ours_num_dic[th] = 0

for iter in range(0, iteration):
  for th in threshold:
    # thにおける各タスクのワーカ候補リスト巡回
    ours_worker_list = []
    baseline_worker_list = []

    for worker_list in ours_all_iter[iter][th].values():
      for worker in worker_list:
        if worker not in ours_worker_list:
          ours_worker_list.append(worker)
    print(ours_worker_list)   
    for worker_list in baseline_all_iter[iter][th].values():
      for worker in worker_list:
        if worker not in baseline_worker_list:
          baseline_worker_list.append(worker)  

    baseline_num_dic[th] += len(baseline_worker_list)
    ours_num_dic[th] += len(ours_worker_list)
print(ours_num_dic)
print(baseline_num_dic) 
# print(baseline_worker_list)
# 割り当て数の平均を求める
for th in threshold:
  baseline_num = baseline_num_dic[th]
  baseline_avg = baseline_num / iteration_time

  ours_num = ours_num_dic[th]
  ours_avg = ours_num / iteration_time

  baseline_num_dic[th] = int(baseline_avg)
  ours_num_dic[th] = int(ours_avg)


#del matplotlib.font_manager.weight_dict['roman']
fig = plt.figure()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.figure(figsize=[6,4])
plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams['font.size'] = 20

'''

'''
# ワーカ能力とタスク困難度の実際の分布
# ヒストグラムを手に入れる
# これはスレッショルド関係ない
# タスク100件, ワーカ全員
# worker_list = list(input_df.columns)
iteration_time = 1
for iteration in range(0, iteration_time):
  print(worker_list)
  sample = devide_sample(task_list, worker_list)
  qualify_task = sample['qualify_task']
  test_task = sample['test_task']
  test_worker = sample['test_worker'] 
  
  baseline_output = make_candidate_all(threshold, input_df, label_df, task_list, worker_list, test_worker, test_task)
  item_param = list(baseline_output[1].values())
  user_param = list(baseline_output[2].values())
print(item_param)


lim = [-4, 4.5]
worker_map = Frequency_Distribution(user_param, lim, class_width=0.5)
item_map = Frequency_Distribution(item_param, lim, class_width=0.5)
print(worker_map)
print(item_map)



bins=np.linspace(-4, 4, 20)
# fig3 = plt.figure()
plt.hist([user_param, item_param], bins, label=['worker', 'task'])
plt.legend(loc='upper left')
plt.xlabel("IRT parameter ")
plt.ylabel("Number of tasks and workers")
# baseline_map.plot.bar(x='階級値', y='度数', label='item', xlabel='difficulty')
# plt.show()



