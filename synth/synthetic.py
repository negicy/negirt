import numpy as np
import pandas as pd
from girth.synthetic import create_synthetic_irt_dichotomous
from girth import onepl_mml
import matplotlib.pyplot as plt 
from scipy.stats import norm
import matplotlib.pyplot as plt 

worker_size = 50
task_size = 100
user_param = {}
item_param = {}

worker = norm.rvs(loc=0, scale=1, size=worker_size)
task = norm.rvs(loc=0.5, scale=1, size=task_size)
a_list = np.array([1]*task_size)

for i in range(0, worker_size):
    worker_id = 'w' + str(i+1)
    user_param[worker_id] = worker[i]

for j in range(0, task_size):
    task_id = 't' + str(j+1)
    item_param[task_id] = task[j]

worker_list = list(user_param.keys())
task_list = list(item_param.keys())

print("============================================-")
difficulty = np.linspace(-2.5, 2.5, 10)
discrimination = np.random.rand(10) + 0.5
theta = np.random.randn(5)
print(discrimination)

# Create Synthetic Data

syn_data = create_synthetic_irt_dichotomous(task, a_list, worker)
# syn_data = create_synthetic_irt_dichotomous(task, discrimination, worker)
syn_df = pd.DataFrame(syn_data, columns=worker_list, index=task_list)

print(syn_df)

# 