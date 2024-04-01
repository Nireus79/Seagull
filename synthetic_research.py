from scipy.stats import bootstrap
import numpy as np
from data_forming import events_data, signal

s = events_data[events_data[signal] == 0]['ret']
b = events_data[events_data[signal] == 1]['ret']

b = (b,)
s = (s,)

# calculate 95% bootstrapped confidence interval for median
bootstrap_cimb = bootstrap(b, np.median, confidence_level=0.95,
                           random_state=52, method='percentile')

# view 95% boostrapped confidence interval
print(bootstrap_cimb.confidence_interval)

bootstrap_cims = bootstrap(s, np.median, confidence_level=0.95,
                           random_state=52, method='percentile')

# view 95% boostrapped confidence interval
print(bootstrap_cims.confidence_interval)
