import matplotlib.pyplot as plt
import numpy as np

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
print(ind)
p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans)

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()

import numpy as np
import matplotlib.pyplot as plt

N = 5
men_means = (20, 35, 30, 35, 27)
women_means = (25, 32, 34, 20, 25)
print('ind:',ind)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()

p1 = ax.bar(ind - width/2, men_means, width, label='Men')
p2 = ax.bar(ind - width/2, women_means, width, bottom=men_means, label='Women')

men_means2 = (15, 25, 20, 30, 22)
women_means2 = (20, 28, 30, 18, 22)

p3 = ax.bar(ind + width/2, men_means2, width, label='Men2')
p4 = ax.bar(ind + width/2, women_means2, width, bottom=men_means2, label='Women2')

ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax.legend()

plt.show()

