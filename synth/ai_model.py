from scipy.stats import norm
import matplotlib.pyplot as plt 
import random
import numpy as np
def ai_model(actual_b, dist):
    b_norm = norm.rvs(loc=actual_b, scale=dist, size=100)

    bins=np.linspace(-1, 1, 20)
    plt.hist([b_norm], bins, label=['task'])
    plt.show()
    return random.choice(b_norm)


test = ai_model(0.5, dist=0.01)
print(test)

