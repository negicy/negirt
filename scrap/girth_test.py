import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous
from girth import twopl_mml, onepl_mml, ability_mle



df = pd.read_csv('input.csv')
df = df.set_index('qid')
# print(df)
task_list = list(df.index)
worker_list = list(df.columns)


data = df.values
print(data)
# Solve for parameters

estimates = twopl_mml(data)

# Unpack estimates
discrimination_estimates = estimates['Discrimination']
difficulty_estimates = estimates['Difficulty']


#print(discrimination_estimates)
# print(difficulty_estimates)

a_list = []
for i in range(100):
    a_list.append(discrimination_estimates)

abilitiy_estimates = ability_mle(data, difficulty_estimates, a_list)
# print(abilitiy_estimates)

user_param = {}
item_param = {}

for k in range(100):
    task_id = task_list[k]
    worker_id = worker_list[k]
    item_param[task_id] = {}
    # item_param[task_id] = difficulty_estimates[k]
    item_param[task_id]['a'] = discrimination_estimates[k]
    # item_param[task_id]['b'] = difficulty_estimates[k]
    user_param[worker_id] = abilitiy_estimates[k] 

print(item_param)
print(user_param)
