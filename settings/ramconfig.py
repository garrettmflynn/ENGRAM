#! /usr/bin/env python
settings = {
    'fs': 2000,
    'all_channels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
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

regions = {
        'laCA3': {
            'channels': {1,2,3,4,5,6},
            'positions': (-28,-10,-22)
        },

        'laCA1': {
            'channels': {7,8,9,10},
            'positions': (-28,-10,-22)
        },

        'raCA3': {
            'channels': {17,18,19,20,21,22},
            'positions': (28,-10,-22)
        },
        
        'raCA1':  {
            'channels': {23,24,25,26},
            'positions': (28,-10,-22)
        },

        'rpCA3':  {
            'channels': {33,34,35,36,37,38},
            'positions': (25,-37,-0)
        },

        'rpCA1':  {
            'channels': {39,40,41,42},
            'positions': (25,-37,-0)
        },
    }