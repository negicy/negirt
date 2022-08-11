from scipy.stats import norm
import matplotlib.pyplot as plt 
import random

def ai_model(actual_b, dist):
    b_norm = norm.rvs(loc=actual_b, scale=1, size=100)
    return random.choice(b_norm)




