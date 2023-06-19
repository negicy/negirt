from old.assignment_method import *
import numpy as np
test_worker = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7"]
test_task = ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7"]

worker_c = {
    "t_1": ["w_4", "w_5", "w_6", "w_7"],
    "t_2": ["w_4", "w_5", "w_6", "w_7"],
    "t_3": ["w_4", "w_5", "w_6", "w_7"],
    "t_4": ["w_5", "w_6", "w_7"],
    "t_5": ["w_6", "w_7"],
    "t_6": ["w_7"],

}

result = assignment(worker_c, test_worker)
print(result)
