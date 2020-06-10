
import numpy as np
import random
from croston import croston
import matplotlib.pyplot as plt


a = np.zeros(50)
val = np.array(random.sample(range(100,200), 10))
idxs = random.sample(range(50), 10)

ts = np.insert(a, idxs, val)


fit_pred = croston.fit_croston(ts, 10, 'original') # croston's method

#fit_pred = croston.fit_croston(ts, 10, 'sba') # Syntetos-Boylan approximation
#fit_pred = croston.fit_croston(ts, 10, 'sbj') # Shale-Boylan-Johnston


yhat = np.concatenate([fit_pred['croston_fittedvalues'], fit_pred['croston_forecast']])

plt.plot(ts)
plt.plot(yhat)

