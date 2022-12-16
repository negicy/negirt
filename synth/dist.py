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
# from survey import *ttrr333232

# 1. worker list, task listをsynthデータで代替する
# 2. IRTによるパラメータ推定は省略

import numpy as np

from girth.synthetic import create_synthetic_irt_dichotomous
from girth import onepl_mml
import matplotlib.pyplot as plt 
from scipy.stats import norm
import matplotlib.pyplot as plt 

worker_size = 100
task_size = 200

user_param_list = norm.rvs(loc=0.90, scale=0.25, size=worker_size)
item_param_list = norm.rvs(loc=0.50, scale=0.25, size=task_size)

bins=np.linspace(-3, 3, 20)
plt.hist([user_param_list, item_param_list], bins, label=['worker', 'task'])
plt.show()