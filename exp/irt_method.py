import math
import girth
import pandas as pd
import numpy as np
import statistics
from girth import twopl_mml, onepl_mml, rasch_mml, ability_mle, ability_map, ability_eap, rasch_jml, onepl_jml,rasch_conditional, onepl_jml
import random
from scipy.stats import norm


def TwoPLM(alpha, beta, theta, d=1.7, c=0.0):
  #p = 1 / (1+math.exp(-d*alpha*(theta-beta)))
  p = c + (1.0 - c) / (1 + math.exp(-(alpha * theta + beta)))
  return p
def OnePLM(beta, theta, d=1.7, c=0.0):
  p = 1 / (1+np.exp(-d*(theta-beta)))
  # p = c + (1.0 - c) / (1 + np.exp(-(1 * theta + beta)))
  return p
def TwoPLM_test(a, b, theta, d=1.7):
  p = 1 / (1+np.exp(-d*a*(theta-b)))
  return p


# input: 01の2D-ndarray, 対象タスクのリスト, 対象ワーカーのリスト
def run_girth_rasch(data, task_list, worker_list):
    #options = {'quadrature_bounds': (-6, 6), 'quadrature_n':10}
    #ßprint(f"task size:{len(data)}")
    options={'max_iteration':1000}
    estimates = rasch_mml(data)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']

    abilitiy_estimates = ability_map(data, difficulty_estimates, discrimination_estimates)
    # print(abilitiy_estimates)

    user_param = {}
    item_param = {}

    for k in range(len(task_list)):
        task_id = task_list[k]

        item_param[task_id] = difficulty_estimates[k]
    for j in range(len(worker_list)):
        worker_id = worker_list[j]
        user_param[worker_id] = abilitiy_estimates[j] 

    return item_param, user_param


def run_girth_onepl(data, task_list, worker_list):
    estimates = onepl_mml(data)
    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']


    a_list = []
    for i in range(len(task_list)):
        a_list.append(discrimination_estimates)

    abilitiy_estimates = ability_mle(data, difficulty_estimates, a_list)
    # print(abilitiy_estimates)
    user_param = {}
    item_param = {}

    for k in range(len(task_list)):
        task_id = task_list[k]
        item_param[task_id] = difficulty_estimates[k]
    for j in range(len(worker_list)):
        worker_id = worker_list[j]
        user_param[worker_id] = abilitiy_estimates[j] 

    return item_param, user_param


# input: 01の2D-ndarray, 対象タスクのリスト, 対象ワーカーのリスト
def run_girth_twopl(data, task_list, worker_list):
    estimates = twopl_mml(data)

    # Unpack estimates(a, b)
    discrimination_estimates = estimates['Discrimination']
    difficulty_estimates = estimates['Difficulty']

    a_list = []
    for i in range(len(task_list)):
        a_list.append(discrimination_estimates)

    #abilitiy_estimates = ability_mle(data, difficulty_estimates, a_list)
    abilitiy_estimates = ability_map(data, difficulty_estimates, a_list, options={'quadrature_bounds': (-4, 4)})
    # print(abilitiy_estimates)

    user_param = {}
    item_param = {}
    discrimination_param = {}

    for k in range(len(task_list)):
        task_id = task_list[k]
        item_param[task_id] = difficulty_estimates[k]
        discrimination_param[task_id] = discrimination_estimates[k]
        #item_param[task_id]['b'] = difficulty_estimates[k]
    for j in range(len(worker_list)):
        worker_id = worker_list[j]
        user_param[worker_id] = abilitiy_estimates[j] 

    return item_param, user_param, discrimination_param
  

def ai_model(actual_b, dist):
  b_norm = norm.rvs(loc=actual_b, scale=dist, size=100)
  # print(b_norm)
  ai_accuracy = list(b_norm).count(actual_b) / len(b_norm)
  # print(ai_accuracy)
  return random.choice(b_norm)

