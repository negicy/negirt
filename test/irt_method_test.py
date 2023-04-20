
import math
import girth
import pandas as pd
import numpy as np
import statistics
from girth import twopl_mml, onepl_mml, rasch_mml, ability_mle
import random

import unittest
import sys
sys.path.append('../exp')

from irt_method import *

#label_df = pd.read_csv("label_df.csv", sep = ",")
#batch_df = pd.read_csv("batch_100.csv", sep = ",")
#label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
#label_df = label_df.set_index('id')

def is_approximately_equal(a, b, rel_tol=1e-9, abs_tol=0.01):
    """
    aとbの相対誤差がrel_tol未満であり、また、絶対誤差がabs_tol未満であれば、近似と判断する。
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class TestIrtFuctions(unittest.TestCase):
    def test_prob(self):
        self.assertEqual(OnePLM(0.5, 0.5), 0.5)
        self.assertTrue(is_approximately_equal(OnePLM(0, 1), 0.8455229))

if __name__ == '__main__':
    unittest.main()