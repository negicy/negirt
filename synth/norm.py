from scipy.stats import norm
import matplotlib.pyplot as plt 
worker = norm.rvs(loc=0, scale=1, size=1000)

task = norm.rvs(loc=0.5, scale=1, size=1000)


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(1, 1, 1)

ax1.hist(task, bins=10)
# ax2.hist(difficulty_est, bins=10)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(1, 1, 1)

ax1.hist(worker, bins=10)
# ax2.hist(difficulty_est, bins=10)  
plt.show()