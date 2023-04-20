import unittest
import numpy as np
import sys
sys.path.append('../exp')
from assignment_method import *


class TestAssignment(unittest.TestCase):
    def test_assignment_1(self):
        test_worker = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7"]
        test_task = ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7"]

        # 各ワーカが日タスクずつ担当できる
        worker_c_1 = {
        "t_1": ["w_2", "w_4", "w_5", "w_6", "w_7"],
        "t_2": ["w_3","w_4", "w_5", "w_6", "w_7"],
        "t_3": ["w_4", "w_5", "w_6", "w_7"],
        "t_4": ["w_5", "w_6", "w_7"],
        "t_5": ["w_6", "w_7"],
        "t_6": ["w_7"],
        "t_7": ["w_1"]
        }
        assign_dic = assignment(worker_c_1, test_worker)
        self.assertTrue(task_variance(assign_dic, test_worker) == 0)
    
    def test_assignment_2(self):
        test_worker = ["w_1", "w_2", "w_3", "w_4", "w_5", "w_6", "w_7", "w_8"]
        test_task = ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7", "w_8"]

        # 各ワーカが日タスクずつ担当できる
        worker_c_2 = {
        "t_1": ["w_1", "w_2", "w_4", "w_5", "w_7"],
        "t_2": ["w_1","w_3","w_4", "w_5", "w_6", "w_7"],
        "t_3": ["w_4", "w_5", "w_6", "w_7"],
        "t_4": ["w_1", "w_5", "w_6", "w_7"],
        "t_5": ["w_6", "w_7"],
        "t_6": ["w_7", "w_5"],
        "t_7": ["w_1", "w_8"],
        "t_8": ["w_6"]
        }
        assign_dic = assignment(worker_c_2, test_worker)
        print(assign_dic)
        self.assertTrue(task_variance(assign_dic, test_worker) == 0)
        

if __name__ == '__main__':
    unittest.main()