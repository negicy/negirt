import sys, os
import numpy as np
import random
import string
import unittest
sys.path.append('../exp')
from simulation import *

# ランダム文字列(workerID, taskID)を生成する関数
def random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class TestSamplingFuctions(unittest.TestCase):
    # 100個のランダム文字列を含む配列を生成する
    worker_list = [random_string(10) for i in range(100)]
    #worker_list = np.random.binomial(10, 0.5, 100)
    task_list = [random_string(10) for i in range(100)]
    sample = devide_sample(task_list, worker_list)
    qualify_task = sample['qualify_task']
    test_task = sample['test_task']
    test_worker = sample['test_worker']

    def test_not_duplocate(self):
        self.assertEqual(len(set(self.qualify_task) & set(self.test_task)),0)
    
    def test_check_size(self):
        test_size = 60
        worker_size = 20
        self.assertEqual(len(self.test_task), test_size)
        self.assertTrue(len(self.test_worker), worker_size)

if __name__ == '__main__':
    unittest.main()