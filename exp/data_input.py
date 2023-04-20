import pickle
import pandas as pd
label_df = pd.read_csv("label_df.csv", sep = ",")
<<<<<<< HEAD
input_df = pd.read_csv("input.csv", sep = ",")
=======
input_df = pd.read_csv("input_no_spam.csv", sep = ",")
>>>>>>> origin/iconference_cr_revision

batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
label_df = label_df.set_index('id')

# origin_id: max(prob_dic)のdict作成
task_dic = {}
for k in range(0, 100):
  task_id = "q"+str(k+1)
  # 元のテストデータのIDを取り出す
  origin_index = 'Input.'+task_id
  origin_id = batch_df[origin_index][0]
  task_dic["q"+str(k+1)] = origin_id

input_df = input_df.set_index('qid')
input_df['task_id'] = 0

q_list = list(input_df.index)

<<<<<<< HEAD

=======
>>>>>>> origin/iconference_cr_revision
# Task IDリストの作成
task_list = list(task_dic.values())

# input_dfのインデックスを置き換え
for q in q_list:
  input_df['task_id'][q] = task_dic[q]
input_df = input_df.set_index('task_id')

worker_list = list(input_df.columns)
print(input_df)

input_data = {'input_df': input_df, 'worker_list': worker_list, 'task_list': task_list}
<<<<<<< HEAD
with open('input_data.pickle', 'wb') as f:
=======
with open('input_data_no_spam.pickle', 'wb') as f:
>>>>>>> origin/iconference_cr_revision
    pickle.dump(input_data, f)