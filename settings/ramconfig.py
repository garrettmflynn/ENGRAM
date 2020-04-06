#! /usr/bin/env python

import numpy as np


metadata = {
    'name': 'keck',
    'extensions': {'signals' : '.ns3', 'events' : '.nex'},
    'project': 'RAM',
    'fs': 2000,
    'all_streams': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    'feature': 'spikes',
    'bandpass_min': 1,
    'bandpass_max': 250,
    '2D_min': 0,
    '2D_max': 150,
    't_bin': .1, # 100 ms is .1 for blackrock
    'f_bin': .5,
    'overlap': .05,
    'norm': True,
    'norm_method':'ZSCORE',
    'log_transform': True,
    'roi':'events',
    'roi_bounds': (-1,1), # Two-second window centered at the event
    'event_of_interest': 'SAMPLE_RESPONSE',
    'model': []
}

# RIGHT = 28
# LEFT = -28
# CA1 = np.array([1.70, 1.60, 1.17])
# CA3 = np.array([1.70, 2.00, 1.85])
# CA1 = CA1/CA1[0]
# CA3 = CA3/CA3[0]
rCA1 = np.array([25,-37,-0]) # CA1*RIGHT
lCA1 = np.array([-25,-37,-0]) # CA1*LEFT
rCA3 = np.array([28,-10,-22]) # CA3*RIGHT
lCA3 = np.array([-28,-10,-22]) # CA3*LEFT
INTERNAL_AP_AXIS = [0,2,0]
rpCA1 = rCA1 - INTERNAL_AP_AXIS
raCA1 = rCA1 + INTERNAL_AP_AXIS
lpCA1 = lCA1 - INTERNAL_AP_AXIS
laCA1 = lCA1 + INTERNAL_AP_AXIS
rpCA3 = rCA3 - INTERNAL_AP_AXIS
raCA3 = rCA3 + INTERNAL_AP_AXIS
lpCA3 = lCA3 - INTERNAL_AP_AXIS
laCA3 = lCA3 + INTERNAL_AP_AXIS

LEVELS = 3

stream_pattern = np.zeros(max(metadata['all_streams'])+1, [('hierarchy', '<U256', LEVELS),\
                        ('positions', np.float32, LEVELS)])

# ___________________________________________ CA3 ___________________________________________

stream_pattern['hierarchy'][[17,18,19,20,21,22]] = ['Right','CA3','Anterior']
stream_pattern['positions'][[17,18,19,20,21,22]] = raCA3

stream_pattern['hierarchy'][[33,34,35,36,37,38]] = ['Right','CA3','Posterior']
stream_pattern['positions'][[33,34,35,36,37,38]] = rpCA3

stream_pattern['hierarchy'][[1,2,3,4,5,6]] = ['Left','CA3','Anterior']
stream_pattern['positions'][[1,2,3,4,5,6]] = laCA3

stream_pattern['hierarchy'][[]] = ['Left','CA3','Posterior']
stream_pattern['positions'][[]] = lpCA3


# ___________________________________________ CA1 ___________________________________________

stream_pattern['hierarchy'][[23,24,25,26]] = ['Right','CA1','Anterior']
stream_pattern['positions'][[23,24,25,26]] = raCA1

stream_pattern['hierarchy'][[39,40,41,42]] = ['Right','CA1','Posterior']
stream_pattern['positions'][[39,40,41,42]] = rpCA1

stream_pattern['hierarchy'][[7,8,9,10]] = ['Left','CA1','Anterior']
stream_pattern['positions'][[7,8,9,10]] = laCA1

stream_pattern['hierarchy'][[]] = ['Left','CA1','Posterior']
stream_pattern['positions'][[]] = lpCA1

stream_pattern = stream_pattern[metadata['all_streams']]

metadata['stream_pattern'] = stream_pattern