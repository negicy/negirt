import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

import statsmodels.formula.api as smf
import statsmodels.api as sm

# sample data
weather = [
    "cloudy", "cloudy",
    "rainy", "rainy",
    "sunny", "sunny"
]

beer = [6, 8, 2, 4, 10, 12]
weather_beer = pd.DataFrame({
    "beer": beer,
    "weather": weather
})
print(weather_beer)


effect = [7, 7, 3, 3, 11, 11]

sns.boxplot(x="weather", y = "beer", data=weather_beer, color="grey")
# plt.show()

mu_effect = sp.mean(effect)
squares_model = sp.mean(effect)
squares_model = sp.sum((effect - mu_effect) ** 2)
print(squares_model)

# 誤差
resid = weather_beer.beer - effect
print(resid)

# 郡内の偏差平方和
squares_resid = sp.sum(resid ** 2)
print(squares_resid)

# 群間・郡内分散

# 群間変動の自由度
df_model = 2
# 郡内変動の自由度
df_resid = 3
# 群間の分散
variance_model = squares_model / df_model
# 郡内の分散
variance_resid = squares_resid /df_resid
print(variance_model)
print(variance_resid)

f_ratio = variance_model / variance_resid
p = 1 - sp.stats.f.cdf(x=f_ratio, dfn=df_model, dfd=df_resid)
print(p)
