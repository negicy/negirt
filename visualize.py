'''
可視化
'''
  # 標準偏差を計算

'''
タスク割り当て結果の可視化
- ワーカ人数
- success assignment rate
- 正解率，分散，トレードオフ
'''

for th in top_assignment_allth:
  res = top_assignment_allth[th]
  print(th, len(res), np.mean(res))

'''
# 割当て結果保存:
'''
import datetime
now = datetime.datetime.now()

# 実験結果を再現できるようにデータを保存
'''
(1)DI,PI,AA,TOP,RANDOMのacc, var, tpの全iterationの平均値
(2)タスクのあるワーカの人数
(3)割り当ての結果正解していたワーカの数
'''
result = {
  'ours_acc': ours_acc, 'top_acc': top_acc, 
  'random_acc': random_acc, 'PI_acc': PI_acc, 'AA_acc': AA_acc,
  'ours_var': ours_var, 'top_var': top_var,  'AA_var': AA_var,
  'random_var': random_var, 'PI_var': PI_var,
  'ours_tp': ours_tp, 'PI_tp': PI_tp, 'AA_tp': AA_tp, 
  'random_tp': random_tp, 'top_tp': top_tp,
  'welldone_dist': welldone_dist, 'worker_with_task': worker_with_task,
  'ours_acc_head': ours_acc_head, 'AA_acc_head': AA_acc_head,
  'ours_acc_tail': ours_acc_tail, 'AA_acc_tail': AA_acc_tail
}


# 結果データの保存
filename = "result/result_{0:%Y%m%d_%H%M%S}.pickle".format(now)
with open(filename, 'wb') as f:
  pickle.dump(result, f)


for th in welldone_dist:
  welldone_dist[th] = welldone_dist[th] / iteration_time

# 割当て成功数ヒストグラム
plt.rcParams["font.size"] = 18
fig =  plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('threshold')
ax.set_ylabel('rate of successful assignments')
ax.bar(['0.5', '0.6', '0.7', '0.8'], welldone_dist.values(), width=0.5, color='forestgreen')
#plt.show()

# タスクのあるワーカ人数をヒストグラムで
# iteration間の平均を求める

num_worker = [[], []]
for th in [0.5, 0.6, 0.7, 0.8]:
  num_worker[0].append(worker_with_task['ours'][th] / iteration_time)
  num_worker[1].append(worker_with_task['AA'][th] / iteration_time)
w = 0.4
y_ours = num_worker[0]
y_AA = num_worker[1]

x1 = [1, 2, 3, 4]
x2 = [1.3, 2.3, 3.3, 4.3]

# 少なくとも1つ以上のタスクを与えられたワーカのヒストグラム
label_x = ['0.5', '0.6', '0.7', '0.8']
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
# 1つ目の棒グラフ
#plt.bar(x1, y_ours, color='blue', width=0.3, label='DI', align="center")

# 2つ目の棒グラフ
#plt.bar(x2, y_AA, color='coral', width=0.3, label='AA', align="center")

# 凡例
plt.xlabel('threshold')
plt.ylabel('Number of workers with tasks')
# X軸の目盛りを置換
plt.xticks([1.15, 2.15, 3.15, 4.15], label_x)
fig.legend(bbox_to_anchor=(0.15, 0.250), loc='upper left')
#plt.show()

# 推移をプロット

result_acc_dic = {
  'ours': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_acc,
  'ours_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_acc_std
  }

result_var_dic = {
  'ours': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_var,
  'ours_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_var_std
}

result_plot_1(threshold, result_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_1(threshold, result_var_dic, ay='variance', bbox=(0.150, 0.800)).show()

# トレードオフのグラフ
ours_trade = tp_acc_plot(ours_tp, ours_acc)
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
ax.set_xlim(0, 30)

bbox=(0.2750, 0.400)
ax.plot(ours_trade[0], ours_trade[1], color='red', label='IRT')
ax.plot(AA_trade[0], AA_trade[1], color='cyan', label='AA')
ax.plot(top_trade[0], top_trade[1], color='blue', label='TOP')
ax.plot(random_trade[0], random_trade[1], color='green', label='RANDOM')
ax.plot(PI_trade[0], PI_trade[1], color='purple', label='IRT(PI)')
fig.legend(bbox_to_anchor=bbox, loc='upper left')
plt.show()


# トレードオフのグラフ
ours_trade = var_acc_plot(ours_var, ours_acc)
AA_trade = var_acc_plot(AA_var, AA_acc)
top_trade = var_acc_plot(top_var, top_acc)
random_trade = var_acc_plot(random_var, random_acc)
PI_trade = var_acc_plot(PI_var, PI_acc)


fig = plt.figure() #親グラフと子グラフを同時に定義
ax1 = fig.add_subplot()
ax1.set_xlabel('Working Opportunity')
ax1.set_ylabel('accuracy')
ax1.set_xlim(0, 30)

bbox=(0.2750, 0.400)
ax1.plot(ours_trade[0], ours_trade[1], color='red', label='IRT')
ax1.plot(AA_trade[0], AA_trade[1], color='cyan', label='AA')
ax1.plot(top_trade[0], top_trade[1], color='blue', label='TOP')
ax1.plot(random_trade[0], random_trade[1], color='green', label='RANDOM')
ax1.plot(PI_trade[0], PI_trade[1], color='purple', label='IRT(PI)')
# ax1.plot(PI_noise1_trade[0], PI_noise1_trade[1], color='orange', label='IRT(PI0.5)')
fig.legend(bbox_to_anchor=bbox, loc='upper left')
plt.show()


# 推移をプロット
plt.rcParams["font.size"] = 22
fig = plt.figure() #親グラフと子グラフを同時に定義
ax = fig.add_subplot()
ax.set_xlabel('threshold')
ax.set_ylabel('margin')
x = np.array(threshold)

ax.plot(x, DI_margin_result, color='red', label='IRT(DI)')
ax.plot(x, PI_margin_result, color='purple', label='IRT(PI)')
#plt.show()

mean_b_tt = np.mean(b_tt_list)
print("平均テストタスク難易度", mean_b_tt)
check_result_parameter_matrix(iteration_time, input_df, PI_all_assign_dic_alliter, DI_all_assign_dic_alliter, full_user_param, full_item_param)

print(num_fit_param/iteration_time)
print(np.mean(NA_count_list)/len(threshold))
