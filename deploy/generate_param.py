# ------------------ noise del hyper param ---------------------
std_scale = [1, 1.05, 1.1, 1.15, 1.]
fp_l = 300       #通過域端周波数[Hz]kotei
fs_l = 1000      #阻止域端周波数[Hz]
gpass_l = 5     #通過域端最大損失[dB]
gstop_l = 40      #阻止域端最小損失[dB]kotei
#L=10000
# ----------------------------------------------------------------
#%%
import numpy as np
import itertools
std_scale = [value for value in np.arange(3, 6, 1)]
fp_l = [value for value in np.arange(100, 700, 200)]
fs_l = [value for value in np.arange(200, 800, 200)]
gpass_l = [value for value in np.arange(5, 6, 1)]
gstop_l = [value for value in np.arange(40, 50, 10)]
print(std_scale)
print(fp_l)
print(fs_l)
print(gpass_l)
print(gstop_l)

# %%
param = list(itertools.product(std_scale, fp_l, fs_l, gpass_l, gstop_l))
print(len(param))
# %%
print(param[1][1])
# %%
