import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import girth
import random
import scipy
import scikit_posthocs as sp
from assignment_method import *
from irt_method import *
from simulation import *
from make_candidate import *

path = os.getcwd()

"""
Real DATA
"""

"""
データ準備
"""

label_df = pd.read_csv("label_df.csv", sep=",")
batch_df = pd.read_csv("batch_100.csv", sep=",")
label_df = label_df.rename(columns={"Unnamed: 0": "id"})
label_df = label_df.set_index("id")

with open("input_data_no_spam.pickle", "rb") as f:
    input_data = pickle.load(f)
    input_df = input_data["input_df"]
    worker_list = input_data["worker_list"]
    task_list = input_data["task_list"]


# 各ワーカーの全体正解率
skill_rate_dic = {}
for w in worker_list:
    correct = 0
    task_num = 0
    for i in task_list:
      task_num += 1
      if input_df[w][i] == 1:
        correct += 1
    skill_rate_dic[w] = (correct / task_num)


# PIの割り当て失敗したタスクidとその回数
underfit_tasks = {}

ours_acc_allth = []
ours_var_allth = []
ours_tp_allth = []
DI_margin_allth = []

DI_onepl_acc_allth = []
DI_onepl_var_allth = []
DI_onepl_tp_allth = []

DI_onepl_margin_acc_allth = []
DI_onepl_margin_var_allth = []
DI_onepl_margin_tp_allth = []

top_acc_allth = []
top_var_allth = []
top_tp_allth = []

AA_acc_allth = []
AA_var_allth = []
AA_tp_allth = []

random_acc_allth = []
random_var_allth = []
random_tp_allth = []

PI_acc_allth = []
PI_var_allth = []
PI_tp_allth = []
PI_margin_allth = []

PI_onepl_acc_allth = []
PI_onepl_var_allth = []
PI_onepl_tp_allth = []

PI_onepl_margin_acc_allth = []
PI_onepl_margin_var_allth = []
PI_onepl_margin_tp_allth = []

PI_all_assign_dic_alliter = {}
DI_all_assign_dic_alliter = {}

PI_onepl_all_assign_dic_alliter = {}
DI_onepl_all_assign_dic_alliter = {}

PI_onepl_margin_all_assign_dic_alliter = {}
DI_onepl_margin_all_assign_dic_alliter = {}


# 0.5, 0.51,...,0.80
threshold=[0.5, 0.6, 0.7, 0.8]
welldone_dist = dict.fromkeys([0.5, 0.6, 0.7, 0.8], 0)

for th in threshold:
    PI_all_assign_dic_alliter[th] = {}
    DI_all_assign_dic_alliter[th] = {}
    PI_onepl_all_assign_dic_alliter[th] = {}
    DI_onepl_all_assign_dic_alliter[th] = {}
    PI_onepl_margin_all_assign_dic_alliter[th] = {}
    DI_onepl_margin_all_assign_dic_alliter[th] = {}

ours_output_alliter = {}
full_output_alliter = {}
worker_list_alliter = {}
task_list_alliter = {}

top_assignment_allth = {}
for th in threshold:
    top_assignment_allth[th] = []

# 承認タスクとテストタスクを分離
# PIでのパラメータ推定
qualify_task = task_list
qualify_dic = {}
for qt in qualify_task:
    qualify_dic[qt] = list(input_df.T[qt])

q_data = np.array(list(qualify_dic.values()))
params_twopl = run_girth_twopl(q_data, task_list, worker_list)

full_item_param = params_twopl[0]
full_user_param = params_twopl[1]

full_user_param_sorted = dict(
    sorted(full_user_param.items(), key=lambda x: x[1], reverse=True)
)
full_item_param_sorted = dict(
    sorted(full_user_param.items(), key=lambda x: x[1], reverse=True)
)

params_onepl = run_girth_onepl(q_data, task_list, worker_list)
full_item_param_onepl = params_onepl[0]
full_user_param_onepl = params_onepl[1]

worker_list_sorted = list(full_user_param_sorted.keys())

b_tt_list = []
num_fit_param = 0
NA_count_list = []

NA_num = 3
"""
イテレーション
"""
print(len(worker_list))
print(len(task_list))
# Solve for parameters
# 割当て結果の比較(random, top, ours)
iteration_time = 200
worker_with_task = {
    "ours": {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0},
    "AA": {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0},
}

for iteration in range(0, iteration_time):
    print("============|", iteration, "|===============")
    ours_acc_perth = []
    ours_var_perth = []
    ours_tp_perth = []
    DI_margin_perth = []

    DI_onepl_acc_perth = []
    DI_onepl_var_perth = []
    DI_onepl_tp_perth = []

    DI_onepl_margin_acc_perth = []
    DI_onepl_margin_var_perth = []
    DI_onepl_margin_tp_perth = []

    top_acc_perth = []
    top_var_perth = []
    top_tp_perth = []

    AA_acc_perth = []
    AA_var_perth = []
    AA_tp_perth = []

    random_acc_perth = []
    random_var_perth = []
    random_tp_perth = []

    PI_acc_perth = []
    PI_var_perth = []
    PI_tp_perth = []
    PI_margin_perth = []

    PI_onepl_acc_perth = []
    PI_onepl_var_perth = []
    PI_onepl_tp_perth = []

    PI_onepl_margin_acc_perth = []
    PI_onepl_margin_var_perth = []
    PI_onepl_margin_tp_perth = []

    sample = devide_sample(task_list, worker_list)
    qualify_task = sample["qualify_task"]
    test_task = sample["test_task"]
    test_worker = sample["test_worker"]

    worker_list_alliter[iteration] = test_worker    
    task_list_alliter[iteration] = test_task

    # test_taskのパラメータ一致度調べる
    num_fit_param += calc_parameter_fit(
        test_task, test_worker, full_item_param, full_user_param, input_df
    )

    # 各手法でのワーカ候補作成
    ours_output = DI_make_candidate_twopl(
        threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task
    )
    ours_candidate = ours_output[0]
    DI_user_param = ours_output[5]
    DI_item_param = ours_output[4]

    DI_onepl_output = DI_make_candidate_onepl(
        threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task
    )
    DI_onepl_candidate = DI_onepl_output[0]
    DI_onepl_user_param = DI_onepl_output[5]
    DI_onepl_item_param = DI_onepl_output[4]

    DI_onepl_margin_output = DI_make_candidate_onepl_margin(
        threshold, input_df, label_df, worker_list, test_worker, qualify_task, test_task
    )
    DI_onepl_margin_candidate = DI_onepl_margin_output[0]

    AA_output = AA_make_candidate(
        threshold, input_df, test_worker, qualify_task, test_task
    )
    AA_candidate = AA_output[0]
    AA_top_workers_dict = AA_output[1]
    AA_worker_rate = AA_output[2]

    PI_output = PI_make_candidate(
        threshold, full_item_param, full_user_param, test_worker, test_task
    )
    PI_candidate = PI_output[0]

    PI_onepl_output = PI_make_candidate(
        threshold, full_item_param_onepl, full_user_param_onepl, test_worker, test_task
    )
    PI_onepl_candidate = PI_onepl_output[0]

    PI_onepl_margin_output = PI_make_candidate_margin(
        threshold, full_item_param_onepl, full_user_param_onepl, test_worker, test_task
    )
    PI_onepl_margin_candidate = PI_onepl_margin_output
    
    top_candidate = top_make_cabdidate(
        threshold, input_df, test_worker, qualify_task, test_task
    )

    # 保存用
    ours_output_alliter[iteration] = ours_output
    full_output_alliter[iteration] = PI_output

    # =======|DIのタスク割り当て|=======
    # worker_c_th = {th: {task: [workers]}}
    for th in ours_candidate:
        candidate_dic = ours_candidate[th]

        DI_assign_dic_opt = {}
        DI_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, DI_user_param
        )

        for worker in DI_assigned:
            for task in DI_assigned[worker]:
                DI_assign_dic_opt[task] = worker
        # print('th'+str(th)+'assignment size'+str(len(assign_dic_opt)))
        # NA タスクをランダム割当て
        '''
        DI_top_workers = sort_test_worker(test_worker, DI_user_param, N=5)
        test_worker_sorted_dict = dict(
            sorted(DI_user_param.items(), key=lambda x: x[1], reverse=True)
        )
        '''
        #test_worker_sorted_list = list(test_worker_sorted_dict.keys())
        #best_worker = test_worker_sorted_list[0]
        DI_sub_workers = extract_sub_worker_irt(
            test_worker, test_task, DI_item_param, DI_user_param
        )
        
        '''
        for task in test_task:
            if task not in DI_assign_dic_opt.keys():
                DI_assign_dic_opt[task] = random.choice(DI_top_workers)
        
        '''
        DI_top_workers = sort_test_worker(test_worker, DI_user_param, N=5)
        
        for task in test_task:
            if task not in DI_assign_dic_opt.keys():
                # 正解確率50%のワーカが一人もいない場合：
                # if DI_item_param[task] > DI_user_param[best_worker]:
                #DI_assign_dic_opt[task] = random.choice(DI_top_workers[10:])
        
                if len(DI_sub_workers[task]) > 0:
                    DI_assign_dic_opt[task] = random.choice(DI_sub_workers[task][:NA_num])
                else:
                    DI_assign_dic_opt[task] = random.choice(test_worker)
        
        # print(len(assign_dic_opt.keys()))
        if th in [0.5, 0.6, 0.7, 0.8]:
            welldone_dist[th] += welldone_count(
                th, DI_assign_dic_opt, full_user_param, full_item_param
            ) / len(test_task)
            # 割当て候補人数を数える
            worker_with_task_list = []
            for worker_set in ours_candidate[th].values():
                for worker in worker_set:
                    if worker not in worker_with_task_list:
                        worker_with_task_list.append(worker)

            worker_with_task["ours"][th] += len(worker_with_task_list)

        # 割り当て結果の精度を求める
        acc = accuracy(DI_assign_dic_opt, input_df)
        DI_all_assign_dic_alliter[th][iteration] = []
        DI_all_assign_dic_alliter[th][iteration].append(DI_assign_dic_opt)
        # print("DI assignment", th, acc, len(assign_dic_opt))

        DI_margin_sum = 0

        for task in DI_assign_dic_opt:
            worker = DI_assign_dic_opt[task]
            DI_margin = (DI_item_param[task] - full_item_param[task]) ** 2
            DI_margin_sum += DI_margin

        DI_mse = math.sqrt(DI_margin_sum / len(DI_assign_dic_opt))
        # 割当て結果分散を求める
        var = task_variance(DI_assign_dic_opt, test_worker)
        # 割当て結果のTPを求める
        tp = calc_tp(DI_assign_dic_opt, test_worker)

        ours_acc_perth.append(acc)
        ours_var_perth.append(var)
        ours_tp_perth.append(tp)
        DI_margin_perth.append(DI_mse)

    ours_acc_allth.append(ours_acc_perth)
    ours_var_allth.append(ours_var_perth)
    ours_tp_allth.append(ours_tp_perth)
    DI_margin_allth.append(DI_margin_perth)

    # =======|DIのタスク割り当て(onepl)|=======
    for th in DI_onepl_candidate:
        candidate_dic = DI_onepl_candidate[th]

        DI_onepl_assign_dic_opt = {}
        DI_onepl_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, DI_onepl_user_param
        )

        for worker in DI_onepl_assigned:
            for task in DI_onepl_assigned[worker]:
                DI_onepl_assign_dic_opt[task] = worker
        # print('th'+str(th)+'assignment size'+str(len(assign_dic_opt)))
        # NA タスクをランダム割当て
        '''
        DI_top_workers = sort_test_worker(test_worker, DI_user_param, N=5)
        test_worker_sorted_dict = dict(
            sorted(DI_user_param.items(), key=lambda x: x[1], reverse=True)
        )
        '''
        #test_worker_sorted_list = list(test_worker_sorted_dict.keys())
        #best_worker = test_worker_sorted_list[0]
        DI_onepl_sub_workers = extract_sub_worker_irt(
            test_worker, test_task, DI_onepl_item_param, DI_onepl_user_param
        )
        DI_onepl_top_workers = sort_test_worker(test_worker, DI_onepl_user_param, N=5)
        
        for task in test_task:
            if task not in DI_onepl_assign_dic_opt.keys():
                # 正解確率50%のワーカが一人もいない場合：
                # if DI_item_param[task] > DI_user_param[best_worker]:
                # DI_assign_dic_opt[task] = random.choice(DI_top_workers[10:])
        
                if len(DI_onepl_sub_workers[task]) > 0:
                    DI_onepl_assign_dic_opt[task] = random.choice(DI_onepl_sub_workers[task][:NA_num])
                else:
                    DI_onepl_assign_dic_opt[task] = random.choice(test_worker)
        
        DI_onepl_all_assign_dic_alliter[th][iteration] = []
        DI_onepl_all_assign_dic_alliter[th][iteration].append(DI_onepl_assign_dic_opt)


        DI_onepl_acc = accuracy(DI_onepl_assign_dic_opt, input_df)
        DI_onepl_var = task_variance(DI_onepl_assign_dic_opt, test_worker)
        DI_onepl_tp = calc_tp(DI_onepl_assign_dic_opt, test_worker)

        DI_onepl_acc_perth.append(DI_onepl_acc)
        DI_onepl_var_perth.append(DI_onepl_var)
        DI_onepl_tp_perth.append(DI_onepl_tp)

    DI_onepl_acc_allth.append(DI_onepl_acc_perth)
    DI_onepl_var_allth.append(DI_onepl_var_perth)
    DI_onepl_tp_allth.append(DI_onepl_tp_perth)

    for th in DI_onepl_margin_candidate:
        candidate_dic = DI_onepl_margin_candidate[th]

        DI_onepl_margin_assign_dic_opt = {}
        DI_onepl_margin_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, full_user_param_onepl
        )

        for worker in DI_onepl_margin_assigned:
            for task in DI_onepl_margin_assigned[worker]:
                DI_onepl_margin_assign_dic_opt[task] = worker

        # NA タスクをランダム割当て
        DI_onepl_margin_top_workers = sort_test_worker(test_worker, full_user_param_onepl, N=5)
        test_worker_sorted_dict = dict(
            sorted(full_user_param_onepl.items(), key=lambda x: x[1], reverse=True)
        )
        DI_onepl_margin_sub_workers = extract_sub_worker_irt(
            test_worker, test_task, DI_onepl_item_param, DI_onepl_user_param
        )

        for task in test_task:
            if task not in DI_onepl_margin_assign_dic_opt.keys():
                # DI_onepl_margin_assign_dic_opt[task] = random.choice(DI_onepl_margin_top_workers)
                
                if len(DI_onepl_margin_sub_workers[task]) > 0:
                    DI_onepl_margin_assign_dic_opt[task] = random.choice(DI_onepl_margin_sub_workers[task][:NA_num])
                else:
                    DI_onepl_margin_assign_dic_opt[task] = random.choice(test_worker)
                
        
        DI_onepl_margin_all_assign_dic_alliter[th][iteration] = []
        DI_onepl_margin_all_assign_dic_alliter[th][iteration].append(DI_onepl_margin_assign_dic_opt)

        DI_onepl_margin_acc = accuracy(DI_onepl_margin_assign_dic_opt, input_df)
        DI_onepl_margin_var = task_variance(DI_onepl_margin_assign_dic_opt, test_worker)
        DI_onepl_margin_tp = calc_tp(DI_onepl_margin_assign_dic_opt, test_worker)

        DI_onepl_margin_acc_perth.append(DI_onepl_margin_acc)
        DI_onepl_margin_var_perth.append(DI_onepl_margin_var)
        DI_onepl_margin_tp_perth.append(DI_onepl_margin_tp)

    DI_onepl_margin_acc_allth.append(DI_onepl_margin_acc_perth)
    DI_onepl_margin_var_allth.append(DI_onepl_margin_var_perth)
    DI_onepl_margin_tp_allth.append(DI_onepl_margin_tp_perth)

    for th in PI_onepl_margin_candidate:
        candidate_dic = PI_onepl_margin_candidate[th]
        PI_onepl_margin_assign_dic_opt = {}

        PI_onepl_margin_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, full_user_param_onepl
        )

        for worker in PI_onepl_margin_assigned:
            for task in PI_onepl_margin_assigned[worker]:
                PI_onepl_margin_assign_dic_opt[task] = worker

        # NA タスクをランダム割当て
        PI_onepl_margin_top_workers = sort_test_worker(test_worker, full_user_param_onepl, N=20)
        test_worker_sorted_dict = dict(
            sorted(full_user_param_onepl.items(), key=lambda x: x[1], reverse=True)
        )
        PI_onepl_margin_sub_workers = extract_sub_worker_irt(
            test_worker, test_task, full_item_param_onepl, full_user_param_onepl
        )

        for task in test_task:
            if task not in PI_onepl_margin_assign_dic_opt.keys():
                #PI_onepl_margin_assign_dic_opt[task] = random.choice(PI_onepl_margin_top_workers)
                
                if len(PI_onepl_margin_sub_workers[task]) > 0:
                    PI_onepl_margin_assign_dic_opt[task] = random.choice(PI_onepl_margin_sub_workers[task][:NA_num])
                else:
                    PI_onepl_margin_assign_dic_opt[task] = random.choice(test_worker)
            
        
        PI_onepl_margin_all_assign_dic_alliter[th][iteration] = []
        PI_onepl_margin_all_assign_dic_alliter[th][iteration].append(PI_onepl_margin_assign_dic_opt)

        PI_onepl_margin_acc = accuracy(PI_onepl_margin_assign_dic_opt, input_df)
        PI_onepl_margin_var = task_variance(PI_onepl_margin_assign_dic_opt, test_worker)
        PI_onepl_margin_tp = calc_tp(PI_onepl_margin_assign_dic_opt, test_worker)

        PI_onepl_margin_acc_perth.append(PI_onepl_margin_acc)
        PI_onepl_margin_var_perth.append(PI_onepl_margin_var)
        PI_onepl_margin_tp_perth.append(PI_onepl_margin_tp)

    PI_onepl_margin_acc_allth.append(PI_onepl_margin_acc_perth)
    PI_onepl_margin_var_allth.append(PI_onepl_margin_var_perth)
    PI_onepl_margin_tp_allth.append(PI_onepl_margin_tp_perth)


    # =======|TOPのタスク割り当て|=======
    for candidate_dic in top_candidate.values():
        index = 0
        top_assign_dic= {}
        for task in candidate_dic:
            index = index % 5
            candidate_list = candidate_dic[task]
            top_assign_dic[task] = random.choice(candidate_list)
            index += 1

        # 割り当て結果の精度を求める
        acc = accuracy(top_assign_dic, input_df)
        var = task_variance(top_assign_dic, test_worker)
        tp = calc_tp(top_assign_dic, test_worker)

        top_acc_perth.append(acc)
        top_var_perth.append(var)
        top_tp_perth.append(tp)

    top_acc_allth.append(top_acc_perth)
    top_var_allth.append(top_var_perth)
    top_tp_allth.append(top_tp_perth)

    # =======|AAのタスク割り当て|=======
    for th in AA_candidate:
        candidate_dic = AA_candidate[th]

        AA_assign_dic_opt = {}

        AA_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, AA_top_workers_dict
        )
        for worker in AA_assigned:
            for task in AA_assigned[worker]:
                AA_assign_dic_opt[task] = worker

        if th in [0.5, 0.6, 0.7, 0.8]:
            # 割当て候補人数を数える
            # worker_with_task['AA'] += len(assign_dic_opt[th])
            worker_with_task_list = []
            for worker_set in AA_candidate[th].values():
                for worker in worker_set:
                    if worker not in worker_with_task_list:
                        worker_with_task_list.append(worker)
            print("重複チェック:", has_duplicates(worker_with_task_list))
            worker_with_task["AA"][th] += len(worker_with_task_list)

        AA_sub_workers = extract_sub_worker_AA(AA_worker_rate)
        AA_top_workers = list(AA_top_workers_dict)

        for task in test_task:
            if task not in AA_assign_dic_opt.keys():
                # if AA_top_workers[0] > 0.5:
                # AA_assign_dic_opt[task] = random.choice(AA_top_workers[:5])
                if len(AA_sub_workers) > 0:
                    AA_assign_dic_opt[task] = random.choice(AA_sub_workers[:NA_num])
                else:
                    AA_assign_dic_opt[task] = random.choice(test_worker)

        # print('割当てサイズ', len(AA_assign_dic_opt))
        # 割り当て結果の精度を求める
        acc = accuracy(AA_assign_dic_opt, input_df)
        var = task_variance(AA_assign_dic_opt, test_worker)
        tp = calc_tp(AA_assign_dic_opt, test_worker)

        AA_acc_perth.append(acc)
        AA_var_perth.append(var)
        AA_tp_perth.append(tp)

    AA_acc_allth.append(AA_acc_perth)
    AA_var_allth.append(AA_var_perth)
    AA_tp_allth.append(AA_tp_perth)

    #  =======|PIのタスク割当て|=======
    for th in PI_candidate:
        candidate_dic = PI_candidate[th]
        PI_assign_dic_opt = {}

        PI_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, full_user_param
        )

        for worker in PI_assigned:
            for task in PI_assigned[worker]:
                PI_assign_dic_opt[task] = worker
        # print(th, len(assign_dic_opt))

        # NA タスクをランダム割当て
        PI_top_workers = sort_test_worker(test_worker, full_user_param, N=20)
        test_worker_sorted_dict = dict(
            sorted(full_user_param.items(), key=lambda x: x[1], reverse=True)
        )
        test_worker_sorted_list = list(test_worker_sorted_dict.keys())
        best_worker = test_worker_sorted_list[0]
        PI_sub_workers = extract_sub_worker_irt(
            test_worker, test_task, full_item_param, full_user_param
        )

        
        for task in test_task:
            if task not in PI_assign_dic_opt.keys():
                # もしthreshold = 0.5でも割当て不可なら
                # if full_item_param[task] > full_user_param[best_worker]:
                #PI_assign_dic_opt[task] = random.choice(PI_top_workers[10:])
    
                if len(PI_sub_workers[task]) > 0:
                    PI_assign_dic_opt[task] = random.choice(PI_sub_workers[task][:NA_num])
                else:
                    PI_assign_dic_opt[task] = random.choice(test_worker)
        

        # 割当て結果の精度を求める
        acc = accuracy(PI_assign_dic_opt, input_df)
        PI_all_assign_dic_alliter[th][iteration] = []
        PI_all_assign_dic_alliter[th][iteration].append(PI_assign_dic_opt)

        PI_margin_sum = 0
        for task in PI_assign_dic_opt:
            worker = PI_assign_dic_opt[task]
            PI_margin = full_user_param[worker] - full_item_param[task]
            PI_margin_sum += PI_margin

        PI_margin_mean = PI_margin_sum / len(PI_assign_dic_opt)

        var = task_variance(PI_assign_dic_opt, test_worker)
        tp = calc_tp(PI_assign_dic_opt, test_worker)

        PI_acc_perth.append(acc)
        PI_var_perth.append(var)
        PI_tp_perth.append(tp)
        PI_margin_perth.append(PI_margin_mean)

    PI_acc_allth.append(PI_acc_perth)
    PI_var_allth.append(PI_var_perth)
    PI_tp_allth.append(PI_tp_perth)
    PI_margin_allth.append(PI_margin_perth)

    for th in PI_onepl_candidate:
        candidate_dic = PI_onepl_candidate[th]
        PI_onepl_assign_dic_opt = {}

        PI_onepl_assigned = optim_assignment(
            candidate_dic, test_worker, test_task, full_user_param_onepl
        )

        for worker in PI_onepl_assigned:
            for task in PI_onepl_assigned[worker]:
                PI_onepl_assign_dic_opt[task] = worker
        # print(th, len(assign_dic_opt))

        # NA タスクをランダム割当て
        PI_onepl_top_workers = sort_test_worker(test_worker, full_user_param_onepl, N=20)
        test_worker_sorted_dict = dict(
            sorted(full_user_param_onepl.items(), key=lambda x: x[1], reverse=True)
        )
        test_worker_sorted_list = list(test_worker_sorted_dict.keys())
        best_worker = test_worker_sorted_list[0]
        PI_onepl_sub_workers = extract_sub_worker_irt(
            test_worker, test_task, full_item_param_onepl, full_user_param_onepl
        )

        
        for task in test_task:
            if task not in PI_onepl_assign_dic_opt.keys():
                # もしthreshold = 0.5でも割当て不可なら
                # if full_item_param[task] > full_user_param[best_worker]:
                #PI_assign_dic_opt[task] = random.choice(PI_top_workers[10:])
    
                if len(PI_onepl_sub_workers[task]) > 0:
                    PI_onepl_assign_dic_opt[task] = random.choice(PI_onepl_sub_workers[task][:NA_num])
                else:
                    PI_onepl_assign_dic_opt[task] = random.choice(test_worker)
        

        # 割当て結果の精度を求める
        PI_onepl_acc = accuracy(PI_onepl_assign_dic_opt, input_df)
        PI_onepl_all_assign_dic_alliter[th][iteration] = []
        PI_onepl_all_assign_dic_alliter[th][iteration].append(PI_onepl_assign_dic_opt)

        PI_onepl_var = task_variance(PI_onepl_assign_dic_opt, test_worker)
        PI_onepl_tp = calc_tp(PI_onepl_assign_dic_opt, test_worker)

        PI_onepl_acc_perth.append(PI_onepl_acc)
        PI_onepl_var_perth.append(PI_onepl_var)
        PI_onepl_tp_perth.append(PI_onepl_tp)
    
    PI_onepl_acc_allth.append(PI_onepl_acc_perth)
    PI_onepl_var_allth.append(PI_onepl_var_perth)
    PI_onepl_tp_allth.append(PI_onepl_tp_perth)


    for th in range(0, len(threshold)):
        assign_dic = random_assignment(test_task, test_worker)
        # 割り当て結果の精度を求める
        acc = accuracy(assign_dic, input_df)
        var = task_variance(assign_dic, test_worker)
        tp = calc_tp(assign_dic, test_worker)

        random_acc_perth.append(acc)
        random_var_perth.append(var)
        random_tp_perth.append(tp)

    random_acc_allth.append(random_acc_perth)
    random_var_allth.append(random_var_perth)
    random_tp_allth.append(random_tp_perth)

"""
割当て結果の計算
"""
ours_acc = [0] * len(threshold)
ours_var = [0] * len(threshold)
ours_tp = [0] * len(threshold)
DI_margin_result = [0] * len(threshold)
ours_acc_std = []
ours_var_std = []

DI_onepl_acc = [0] * len(threshold)
DI_onepl_var = [0] * len(threshold)
DI_onepl_tp = [0] * len(threshold)
DI_onepl_acc_std = []
DI_onepl_var_std = []

DI_onepl_margin_acc = [0] * len(threshold)
DI_onepl_margin_var = [0] * len(threshold)
DI_onepl_margin_tp = [0] * len(threshold)
DI_onepl_margin_acc_std = []
DI_onepl_margin_var_std = []

PI_onepl_margin_acc = [0] * len(threshold)
PI_onepl_margin_var = [0] * len(threshold)
PI_onepl_margin_tp = [0] * len(threshold)
PI_onepl_margin_acc_std = []
PI_onepl_margin_var_std = []

top_acc = [0] * len(threshold)
top_var = [0] * len(threshold)
top_tp = [0] * len(threshold)
top_acc_std = []
top_var_std = []

AA_acc = [0] * len(threshold)
AA_var = [0] * len(threshold)
AA_tp = [0] * len(threshold)
AA_acc_std = []
AA_var_std = []

random_acc = [0] * len(threshold)
random_var = [0] * len(threshold)
random_tp = [0] * len(threshold)
random_acc_std = []
random_var_std = []

PI_acc = [0] * len(threshold)
PI_var = [0] * len(threshold)
PI_tp = [0] * len(threshold)
PI_acc_std = []
PI_var_std = []
PI_margin_result = [0] * len(threshold)

PI_onepl_acc = [0] * len(threshold)
PI_onepl_var = [0] * len(threshold)
PI_onepl_tp = [0] * len(threshold)

ours_result = combine_iteration(
    threshold, iteration_time, ours_acc_allth, ours_var_allth, ours_tp_allth
)
ours_acc = ours_result[0]
ours_var = ours_result[1]
ours_tp = ours_result[2]
# 標準偏差を計算
ours_acc_std = ours_result[3]
ours_var_std = ours_result[4]
ours_acc_head = ours_result[5]
ours_acc_tail = ours_result[6]

for th in range(0, len(threshold)):
    DI_margin_sum_th = 0
    for i in range(0, iteration_time):
        DI_margin_sum_th += DI_margin_allth[i][th]
    # acc, var, tpの平均を計算
    DI_margin_result[th] = DI_margin_sum_th / iteration_time
# DI_margin_result[th] = DI_margin_sum_th / iteration_time

print(DI_onepl_acc_allth)
DI_onepl_result = combine_iteration(
    threshold, iteration_time, DI_onepl_acc_allth, DI_onepl_var_allth, DI_onepl_tp_allth
)
DI_onepl_acc = DI_onepl_result[0]
DI_onepl_var = DI_onepl_result[1]
DI_onepl_tp = DI_onepl_result[2]
# 標準偏差を計算
DI_onepl_acc_std = DI_onepl_result[3]
DI_onepl_var_std = DI_onepl_result[4]
DI_onepl_acc_head = DI_onepl_result[5]
DI_onepl_acc_tail = DI_onepl_result[6]

DI_onepl_margin_result = combine_iteration(
    threshold, iteration_time, DI_onepl_margin_acc_allth, DI_onepl_margin_var_allth, DI_onepl_margin_tp_allth
)
DI_onepl_margin_acc = DI_onepl_margin_result[0]
DI_onepl_margin_var = DI_onepl_margin_result[1]
DI_onepl_margin_tp = DI_onepl_margin_result[2]
# 標準偏差を計算
DI_onepl_margin_acc_std = DI_onepl_margin_result[3]
DI_onepl_margin_var_std = DI_onepl_margin_result[4]
DI_onepl_margin_acc_head = DI_onepl_margin_result[5]
DI_onepl_margin_acc_tail = DI_onepl_margin_result[6]

PI_onepl_margin_result = combine_iteration(
    threshold, iteration_time, PI_onepl_margin_acc_allth, PI_onepl_margin_var_allth, PI_onepl_margin_tp_allth
)
PI_onepl_margin_acc = PI_onepl_margin_result[0]
PI_onepl_margin_var = PI_onepl_margin_result[1]
PI_onepl_margin_tp = PI_onepl_margin_result[2]
# 標準偏差を計算
PI_onepl_margin_acc_std = PI_onepl_margin_result[3]
PI_onepl_margin_var_std = PI_onepl_margin_result[4]
PI_onepl_margin_acc_head = PI_onepl_margin_result[5]
PI_onepl_margin_acc_tail = PI_onepl_margin_result[6]

top_result = combine_iteration(
    threshold, iteration_time, top_acc_allth, top_var_allth, top_tp_allth
)
top_acc = top_result[0]
top_var = top_result[1]
top_tp = top_result[2]
# 標準偏差を計算
top_acc_std = top_result[3]
top_var_std = top_result[4]
top_acc_head = top_result[5]
top_acc_tail = top_result[6]

AA_result = combine_iteration(
    threshold, iteration_time, AA_acc_allth, AA_var_allth, AA_tp_allth
)
AA_acc = AA_result[0]
AA_var = AA_result[1]
AA_tp = AA_result[2]
# 標準偏差を計算
AA_acc_std = AA_result[3]
AA_var_std = AA_result[4]
AA_acc_head = AA_result[5]
AA_acc_tail = AA_result[6]

random_result = combine_iteration(
    threshold, iteration_time, random_acc_allth, random_var_allth, random_tp_allth
)
random_acc = random_result[0]
random_var = random_result[1]
random_tp = random_result[2]

# 標準偏差を計算
random_acc_std = random_result[3]
random_var_std = random_result[4]
# print(random_acc)

PI_result = combine_iteration(
    threshold, iteration_time, PI_acc_allth, PI_var_allth, PI_tp_allth
)
PI_acc = PI_result[0]
PI_var = PI_result[1]
PI_tp = PI_result[2]
# 標準偏差を計算
PI_acc_std = PI_result[3]
PI_var_std = top_result[4]
PI_acc_head = PI_result[5]
PI_acc_tail = PI_result[6]

for th in range(0, len(threshold)):
    PI_margin_sum_th = 0
    for i in range(0, iteration_time):
        PI_margin_sum_th += PI_margin_allth[i][th]
    # acc, var, tpの平均を計算
    PI_margin_result[th] = PI_margin_sum_th / iteration_time

PI_onepl_result = combine_iteration(
    threshold, iteration_time, PI_onepl_acc_allth, PI_onepl_var_allth, PI_onepl_tp_allth
)
PI_onepl_acc = PI_onepl_result[0]
PI_onepl_var = PI_onepl_result[1]
PI_onepl_tp = PI_onepl_result[2]
# 標準偏差を計算
PI_onepl_acc_std = PI_onepl_result[3]
PI_onepl_var_std = PI_onepl_result[4]
PI_onepl_acc_head = PI_onepl_result[5]
PI_onepl_acc_tail = PI_onepl_result[6]


"""
可視化
"""
# パラメータの関係と正誤の関係
res_twopl = check_result_parameter_matrix(iteration_time, input_df, PI_all_assign_dic_alliter, DI_all_assign_dic_alliter, full_user_param, full_item_param)
PI_twopl_res_dic = res_twopl[0]
DI_twopl_res_dic = res_twopl[1]

res_onepl = check_result_parameter_matrix(iteration_time, input_df, PI_onepl_all_assign_dic_alliter, DI_onepl_all_assign_dic_alliter, full_user_param_onepl, full_item_param_onepl)
PI_onepl_res_dic = res_onepl[0]
DI_onepl_res_dic = res_onepl[1]

res_onepl_margin = check_result_parameter_matrix(iteration_time, input_df, PI_onepl_margin_all_assign_dic_alliter, DI_onepl_margin_all_assign_dic_alliter, full_user_param_onepl, full_item_param_onepl)
PI_onepl_margin_res_dic = res_onepl_margin[0]
DI_onepl_margin_res_dic = res_onepl_margin[1]

res_worker = check_result_worker_parameter(iteration_time, input_df, PI_all_assign_dic_alliter, DI_all_assign_dic_alliter, full_user_param, full_item_param,  worker_list_alliter)
worker_rank_dict_PI = res_worker[0]
worker_rank_dict_DI = res_worker[1]

res_worker_onepl = check_result_worker_parameter(iteration_time, input_df, PI_onepl_all_assign_dic_alliter, DI_onepl_all_assign_dic_alliter, full_user_param_onepl, full_item_param_onepl,  worker_list_alliter)
worker_rank_dict_PI_onepl = res_worker_onepl[0]
worker_rank_dict_DI_onepl = res_worker_onepl[1]

res_worker_onepl_margin = check_result_worker_parameter(iteration_time, input_df, PI_onepl_margin_all_assign_dic_alliter, DI_onepl_margin_all_assign_dic_alliter, full_user_param_onepl, full_item_param_onepl,  worker_list_alliter)
worker_rank_dict_PI_onepl_margin = res_worker_onepl_margin[0]
worker_rank_dict_DI_onepl_margin = res_worker_onepl_margin[1]

# worker_rank_dictをヒストグラムで可視化
# ヒストグラム描画: 横軸: threshold, 縦軸: θ < bで正答したワーカ数

print(f"PI variance:{PI_onepl_var}")
for th in threshold:
    ind = np.array(worker_rank_dict_PI_onepl[th].keys())

    assigned_ot_ut = worker_rank_dict_PI_onepl[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    plt.bar(keys, ot_values, color='purple', label='ot')
    plt.bar(keys, ut_values, color='violet', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')

    # Labels and title
    plt.xlabel('Key')
    plt.ylabel('Values')
    plt.title(f'Stacked Histogram of OT and UT values(PI-1PLM@{th})')
    plt.legend()

    # Show the plot
    plt.show()

print(f"DI variance:{DI_onepl_var}")
for th in threshold:

    ind = np.array(worker_rank_dict_DI_onepl[th].keys())

    assigned_ot_ut = worker_rank_dict_DI_onepl[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    # 積み上げグラフとして表示：4段
    plt.bar(keys, ot_values, color='red', label='ot')
    plt.bar(keys, ut_values, color='orange', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')


    # Labels and title
    plt.xlabel('Key')
    plt.ylabel('Values')
    plt.title(f'Stacked Histogram of OT and UT values(DI-1PLM@{th})')
    plt.legend()

    # Show the plot
    plt.show()


print(f"PI variance:{PI_onepl_margin_var}")
for th in threshold:
    ind = np.array(worker_rank_dict_PI_onepl_margin[th].keys())

    assigned_ot_ut = worker_rank_dict_PI_onepl_margin[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    plt.bar(keys, ot_values, color='purple', label='ot')
    plt.bar(keys, ut_values, color='violet', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')

    # Labels and title
    plt.xlabel('Key')
    plt.ylabel('Values')
    plt.title(f'Stacked Histogram of OT and UT values(PI-1PLM@{th})')
    plt.legend()

    # Show the plot
    plt.show()

print(f"DI variance:{DI_onepl_margin_var}")
for th in threshold:
    ind = np.array(worker_rank_dict_DI_onepl_margin[th].keys())

    assigned_ot_ut = worker_rank_dict_DI_onepl_margin[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    # 積み上げグラフとして表示：4段
    plt.bar(keys, ot_values, color='red', label='ot')
    plt.bar(keys, ut_values, color='orange', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')


    # Labels and title
    plt.xlabel('Key')
    plt.ylabel('Values')
    plt.title(f'Stacked Histogram of OT and UT values(DI-1PLM+Margin@{th})')
    plt.legend()

    # Show the plot
    plt.show()


print(f"PI variance:{PI_var}")
for th in threshold:
    print(PI_var)
    ind = np.array(worker_rank_dict_PI[th].keys())

    assigned_ot_ut = worker_rank_dict_PI[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    plt.bar(keys, ot_values, color='purple', label='ot')
    plt.bar(keys, ut_values, color='violet', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')

    # Labels and title
    plt.xlabel('Key')
    plt.ylabel('Values')
    plt.title(f'Stacked Histogram of OT and UT values(PI-2PLM@{th})')
    plt.legend()

    # Show the plot
    plt.show()

print(f"DI variance:{ours_var}")
for th in threshold:

    ind = np.array(worker_rank_dict_DI[th].keys())

    assigned_ot_ut = worker_rank_dict_DI[th]
    # Extracting 'ot' and 'ut' values
    ot_values = [assigned_ot_ut[key]['ot'] for key in assigned_ot_ut]
    ut_values = [assigned_ot_ut[key]['ut'] for key in assigned_ot_ut]
    of_values = [assigned_ot_ut[key]['of'] for key in assigned_ot_ut]
    uf_values = [assigned_ot_ut[key]['uf'] for key in assigned_ot_ut]

    # Keys for the x-axis
    keys = list(assigned_ot_ut.keys())

    # Plotting
    # 積み上げグラフとして表示：4段
    plt.bar(keys, ot_values, color='red', label='ot')
    plt.bar(keys, ut_values, color='orange', bottom=ot_values, label='ut')
    plt.bar(keys, of_values, color='green', bottom=[v1 + v2 for v1, v2 in zip(ot_values, ut_values)], label='of')
    plt.bar(keys, uf_values, color='blue', bottom=[v1 + v2 +v3 for v1, v2, v3 in zip(ot_values, ut_values, of_values)], label='uf')

    # Labels and title
    plt.xlabel('Key')
    plt.ylabel('Values')
    plt.title(f'Stacked Histogram of OT and UT values(2PLM@{th})')
    plt.legend()
    # Show the plot
    plt.show()

# ヒストグラム描画: 横軸: threshold, 縦軸: θ < bで正答したタスク数
# パラメータの関係と正誤の関係 1PLM
DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

for th in threshold:
    DI_ut_task.append(DI_onepl_res_dic[th][1])
    PI_ut_task.append(PI_onepl_res_dic[th][1])

    DI_ot_task.append(DI_onepl_res_dic[th][0])
    PI_ot_task.append(PI_onepl_res_dic[th][0])

    PI_of_task.append(PI_onepl_res_dic[th][2])
    DI_of_task.append(DI_onepl_res_dic[th][2])

    PI_uf_task.append(DI_onepl_res_dic[th][3])
    DI_uf_task.append(DI_onepl_res_dic[th][3])

ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, DI_ot_task, width, color='red')
p2 = ax.bar(ind - width/2, DI_ut_task, width, bottom=DI_ot_task, color='orange')
p3 = ax.bar(ind + width/2, PI_ot_task, width,  color='purple')
p4 = ax.bar(ind + width/2, PI_ut_task, width, bottom=PI_ot_task, color='violet')
#p1 = ax.bar(ind - width/2, PI_ot_task, width, color='red')
#p3 = ax.bar(ind + width/2, PI_of_task, width,  color='purple')
plt.ylabel('Number of tasks')
plt.title('Number of correctly answered task by DI,PI(1PLM)')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
plt.yticks(np.arange(0, 51, 5))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()

# パラメータの関係と正誤の関係 1PLM+Margin
DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

for th in threshold:
    DI_ot_task.append(DI_onepl_margin_res_dic[th][0])
    PI_ot_task.append(PI_onepl_margin_res_dic[th][0])

    DI_ut_task.append(DI_onepl_margin_res_dic[th][1])
    PI_ut_task.append(PI_onepl_margin_res_dic[th][1])

    PI_of_task.append(PI_onepl_margin_res_dic[th][2])
    DI_of_task.append(DI_onepl_margin_res_dic[th][2])

    PI_uf_task.append(DI_onepl_margin_res_dic[th][3])
    DI_uf_task.append(DI_onepl_margin_res_dic[th][3])

ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, DI_ot_task, width, color='red')
p2 = ax.bar(ind - width/2, DI_ut_task, width, bottom=DI_ot_task, color='orange')
p3 = ax.bar(ind + width/2, PI_ot_task, width,  color='purple')
p4 = ax.bar(ind + width/2, PI_ut_task, width, bottom=PI_ot_task, color='violet')
#p1 = ax.bar(ind - width/2, PI_ot_task, width, color='red')
#p3 = ax.bar(ind + width/2, PI_of_task, width,  color='purple')
plt.ylabel('Number of tasks')
plt.title('Number of correctly answered task by DI,PI(1PLM+Margin)')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
plt.yticks(np.arange(0, 51, 5))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()

# パラメータの関係と正誤の関係 2PLM
DI_ut_task = []
PI_ut_task = []
DI_ot_task = []
PI_ot_task = []
PI_of_task = []
DI_of_task = []
PI_uf_task = []
DI_uf_task = []

for th in threshold:
    DI_ot_task.append(DI_twopl_res_dic[th][0])
    PI_ot_task.append(PI_twopl_res_dic[th][0])

    DI_ut_task.append(DI_twopl_res_dic[th][1])
    PI_ut_task.append(PI_twopl_res_dic[th][1])

    PI_of_task.append(PI_twopl_res_dic[th][2])
    DI_of_task.append(DI_twopl_res_dic[th][2])

    PI_uf_task.append(DI_twopl_res_dic[th][3])
    DI_uf_task.append(DI_twopl_res_dic[th][3])

ind = np.arange(len(threshold))  # the x locations for the groups
width = 0.0275 * 12      # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
p1 = ax.bar(ind - width/2, DI_ot_task, width, color='red')
p2 = ax.bar(ind - width/2, DI_ut_task, width, bottom=DI_ot_task, color='orange')
p3 = ax.bar(ind + width/2, PI_ot_task, width,  color='purple')
p4 = ax.bar(ind + width/2, PI_ut_task, width, bottom=PI_ot_task, color='violet')
#p1 = ax.bar(ind - width/2, PI_ot_task, width, color='red')
#p3 = ax.bar(ind + width/2, PI_of_task, width,  color='purple')
plt.ylabel('Number of tasks')
plt.title('Number of correctly answered task by DI,PI(2PLM)')
plt.xticks(ind, ('0.5', '0.6', '0.7', '0.8'))
plt.yticks(np.arange(0, 51, 5))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()


# 標準偏差を計算
"""
タスク割り当て結果の可視化
- ワーカ人数
- success assignment rate
- 正解率，分散，トレードオフ
"""
for th in top_assignment_allth:
    res = top_assignment_allth[th]
    print(th, len(res), np.mean(res))

"""
# 割当て結果保存:
"""
import datetime

now = datetime.datetime.now()

# 実験結果を再現できるようにデータを保存
"""
(1)DI,PI,AA,TOP,RANDOMのacc, var, tpの全iterationの平均値
(2)タスクのあるワーカの人数
(3)割り当ての結果正解していたワーカの数
"""
result = {
    "ours_acc": ours_acc,
    "DI_onepl_acc": DI_onepl_acc,
    "DI_onepl_margin_acc": DI_onepl_margin_acc,
    "PI_onepl_margin_acc": PI_onepl_margin_acc,
    "top_acc": top_acc,
    "random_acc": random_acc,
    "PI_acc": PI_acc,
    "AA_acc": AA_acc,
    "PI_onepl_acc": PI_onepl_acc,

    "ours_var": ours_var,
    "DI_onepl_var": DI_onepl_var,
    "DI_onepl_margin_var": DI_onepl_margin_var,
    "PI_onepl_margin_var": PI_onepl_margin_var,
    "top_var": top_var,
    "AA_var": AA_var,
    "random_var": random_var,
    "PI_var": PI_var,
    "PI_onepl_var": PI_onepl_var,

    "ours_tp": ours_tp,
    "DI_onepl_tp": DI_onepl_tp,
    "DI_onepl_margin_tp": DI_onepl_margin_tp,
    "PI_onepl_margin_tp": PI_onepl_margin_tp,
    "PI_tp": PI_tp,
    "AA_tp": AA_tp,
    "random_tp": random_tp,
    "top_tp": top_tp,
    "PI_onepl_tp": PI_onepl_tp,

    "welldone_dist": welldone_dist,
    "worker_with_task": worker_with_task,

    "ours_acc_head": ours_acc_head,
    "DI_onepl_acc_head": DI_onepl_acc_head,
    "DI_onepl_margin_acc_head": DI_onepl_margin_acc_head,
    "AA_acc_head": AA_acc_head,
    "top_acc_head": top_acc_head,
    #"PI_acc_head": PI_acc_head,
    "PI_onepl_acc_head": PI_onepl_acc_head,
    "PI_onepl_margin_acc_head": PI_onepl_margin_acc_head,

    "ours_acc_tail": ours_acc_tail,
    "DI_onepl_acc_tail": DI_onepl_acc_tail,
    "AA_acc_tail": AA_acc_tail,
    "top_acc_tail": top_acc_tail,
    "PI_acc_tail": PI_acc_tail,
    "PI_onepl_acc_tail": PI_onepl_acc_tail,

    "ours_acc_std": ours_acc_std,
    "DI_onepl_acc_std": DI_onepl_acc_std,
    "DI_onepl_margin_acc_std": DI_onepl_margin_acc_std,
    "top_acc_std": top_acc_std,
    "AA_acc_std": AA_acc_std,
    "random_acc_std": random_acc_std,
    "PI_acc_std": PI_acc_std,
    "PI_onepl_acc_std": PI_onepl_acc_std,
    "PI_onepl_margin_acc_std": PI_onepl_margin_acc_std,

    "ours_var_std": ours_var_std,
    "DI_onepl_var_std": DI_onepl_var_std,
    "DI_onepl_margin_var_std": DI_onepl_margin_var_std,
    "top_var_std": top_var_std,
    "AA_var_std": AA_var_std,
    "random_var_std": random_var_std,
    "PI_var_std": PI_var_std,
    "PI_onepl_var_std": PI_onepl_var_std,
    "PI_onepl_margin_var_std": PI_onepl_margin_var_std,

    "worker_rank_dict_PI": worker_rank_dict_PI,
    "worker_rank_dict_DI": worker_rank_dict_DI,

    "worker_rank_dict_PI_onepl": worker_rank_dict_PI_onepl,
    "worker_rank_dict_DI_onepl": worker_rank_dict_DI_onepl,

    "worker_rank_dict_PI_onepl_margin": worker_rank_dict_PI_onepl_margin,
    "worker_rank_dict_DI_onepl_margin": worker_rank_dict_DI_onepl_margin,

    "worker_rank_dict_PI_onepl_margin": worker_rank_dict_PI_onepl_margin,
    "worker_rank_dict_DI_onepl_margin": worker_rank_dict_DI_onepl_margin,

    "PI_onepl_res_dic": PI_onepl_res_dic,
    "DI_onepl_res_dic": DI_onepl_res_dic,
    "PI_twopl_res_dic": PI_twopl_res_dic,
    "DI_twopl_res_dic": DI_twopl_res_dic,
    "PI_onepl_margin_res_dic": PI_onepl_margin_res_dic,
    "DI_onepl_margin_res_dic": DI_onepl_margin_res_dic,

    "PI_all_assign_dic_alliter": PI_all_assign_dic_alliter,
    "DI_all_assign_dic_alliter": DI_all_assign_dic_alliter,
    "DI_onepl_margin_all_assign_dic_alliter": DI_onepl_margin_all_assign_dic_alliter,
    "PI_onepl_all_assign_dic_alliter": PI_onepl_all_assign_dic_alliter,
    "DI_onepl_all_assign_dic_alliter": DI_onepl_all_assign_dic_alliter,
    "PI_onepl_margin_all_assign_dic_alliter": PI_onepl_margin_all_assign_dic_alliter,

    "worker_list_alliter": worker_list_alliter,
    "task_list_alliter": task_list_alliter,

    "PI_acc_allth": PI_acc_allth,
    "DI_acc_allth": ours_acc_allth,
    "PI_onepl_acc_allth": PI_onepl_acc_allth,
    "DI_onepl_acc_allth": DI_onepl_acc_allth,
    "PI_onepl_margin_acc_allth": PI_onepl_margin_acc_allth,
    "DI_onepl_margin_acc_allth": DI_onepl_margin_acc_allth,
    "top_acc_allth": top_acc_allth,
    "random_acc_allth": random_acc_allth,
    "AA_acc_allth": AA_acc_allth,

}

# 結果データの保存
filename = "results/result_{0:%Y%m%d_%H%M%S}.pickle".format(now)
with open(filename, "wb") as f:
    pickle.dump(result, f)

# タスクのあるワーカ人数をヒストグラムで
histgram_worker_with_task(worker_with_task, iteration_time)

# 割当て成功数ヒストグラム
histgram_welldone(welldone_dist, iteration_time)

# パラメータと結果正誤の関係
# 推移をプロット

# onePLMの結果をプロット
result_onepl_acc_dic = {
  'DI': DI_onepl_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_onepl_acc,
  'DI_std': DI_onepl_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_onepl_acc_std
  }

result_onepl_var_dic = {
  'DI': DI_onepl_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_onepl_var,
  'DI_std': DI_onepl_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_onepl_var_std
}

result_onepl_tp_dic = {
  'DI': DI_onepl_tp, 'top': top_tp, 'AA': AA_tp, 'random': random_tp, 'PI': PI_onepl_tp
}



result_plot_acc_var(threshold, result_onepl_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_acc_var(threshold, result_onepl_var_dic, ay='variance', bbox=(0.150, 0.800)).show()

result_plot_tradeoff(result_onepl_tp_dic, result_onepl_acc_dic).show()
result_plot_tradeoff(result_onepl_var_dic, result_onepl_acc_dic).show()

# onePLM+marginの結果をプロット
result_onepl_margin_acc_dic = {
  'DI': DI_onepl_margin_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_onepl_margin_acc,
  'DI_std': DI_onepl_margin_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_onepl_margin_acc_std
  }

result_onepl_margin_var_dic = {
  'DI': DI_onepl_margin_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_onepl_margin_var,
  'DI_std': DI_onepl_margin_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_onepl_margin_var_std
}

result_onepl_margin_tp_dic = {
  'DI': DI_onepl_margin_tp, 'top': top_tp, 'AA': AA_tp, 'random': random_tp, 'PI': PI_onepl_margin_tp
}

result_plot_acc_var(threshold, result_onepl_margin_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_acc_var(threshold, result_onepl_margin_var_dic, ay='variance', bbox=(0.150, 0.800)).show()
result_plot_tradeoff(result_onepl_margin_tp_dic, result_onepl_margin_acc_dic).show()
result_plot_tradeoff(result_onepl_margin_var_dic, result_onepl_margin_acc_dic).show()

'''

# twoPLMの結果をプロット
result_acc_dic = {
  'DI': ours_acc, 'top': top_acc, 'AA': AA_acc, 'random': random_acc, 'PI': PI_acc,
  'DI_std': ours_acc_std, 'top_std': top_acc_std, 'AA_std': AA_acc_std, 'random_std': random_acc_std, 'PI_std': PI_acc_std
  }

result_var_dic = {
  'DI': ours_var, 'top': top_var, 'AA': AA_var, 'random': random_var, 'PI': PI_var,
  'DI_std': ours_var_std, 'top_std': top_var_std, 'AA_std': AA_var_std, 'random_std': random_var_std, 'PI_std': PI_var_std
}

result_tp_dic = {
  'DI': ours_tp, 'top': top_tp, 'AA': AA_tp, 'random': random_tp, 'PI': PI_tp
}

result_plot_acc_var(threshold, result_acc_dic, ay='accuracy', bbox=(0.150, 0.400)).show()
result_plot_acc_var(threshold, result_var_dic, ay='variance', bbox=(0.150, 0.800)).show()
result_plot_tradeoff(result_tp_dic, result_acc_dic).show()
result_plot_tradeoff(result_var_dic, result_acc_dic).show()

'''