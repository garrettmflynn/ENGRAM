#! /usr/bin/env python

metadata = {
    'name': 'X', # Name of User
    'extensions': {'signals' : '.ns3', 'events' : '.nex'}, # Data File Extensions
    'project': 'Y', # Name of Project
    'fs': 2000, # Sampling Frequency
    'all_streams': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42], # Independent Data Stream (or Channels)
    'feature': 'spikes', # Feature to Use for Machine Learning
    'bandpass_min': 1, # Minimum LFP Bandpass Frequency 
    'bandpass_max': 250, # Maximum LFP Bandpass Frequency 
    '2D_min': 0, # Minimum Frequency of Interest
    '2D_max': 150, # Maximum Frequency of Interest 
    't_bin': .1, # In Seconds (for STFT)
    'f_bin': .5, # In Hz (for STFT)
    'overlap': .05, # For STFT
    'norm': True, # Normalization (binary choice)
    'norm_method':'ZSCORE', # Normalization (method)
    'log_transform': True, # Option to Log-Transform Your Data Before Normalization (binary choice)
    'roi':'events', # Method of choosing your ROI (either 'events' or 'trials')
    'roi_bounds': (-1,1), # In seconds centered around the event of interest
    'event_of_interest': 'SAMPLE_RESPONSE', # Must be a label in your events file
    'model': [] # If desired for ML
}

CA1_OFFSET = 5

LEVELS = 3

stream_pattern = np.zeros(max(metadata['all_streams'])+1, [('hierarchy', '<U256', LEVELS),\
                        ('positions', np.float32, LEVELS)])

# ___________________________________________ CA3 ___________________________________________

stream_pattern['hierarchy'][[17,18,19,20,21,22]] = ['Right','CA3','Anterior']
stream_pattern['positions'][[17,18,19,20,21,22]] = [28,-10,-22]

stream_pattern['hierarchy'][[33,34,35,36,37,38]] = ['Right','CA3','Posterior']
stream_pattern['positions'][[33,34,35,36,37,38]] = [25,-37,-0]

stream_pattern['hierarchy'][[1,2,3,4,5,6]] = ['Left','CA3','Anterior']
stream_pattern['positions'][[1,2,3,4,5,6]] = [-28,-10,-22]

stream_pattern['hierarchy'][[]] = ['Left','CA3','Posterior']
stream_pattern['positions'][[]] = [-25,-37,-0]


# ___________________________________________ CA1 ___________________________________________

stream_pattern['hierarchy'][[23,24,25,26]] = ['Right','CA1','Anterior']
stream_pattern['positions'][[23,24,25,26]] = [28+CA1_OFFSET,-10,-22]

stream_pattern['hierarchy'][[39,40,41,42]] = ['Right','CA1','Posterior']
stream_pattern['positions'][[39,40,41,42]] = [25+CA1_OFFSET,-37,-0]

stream_pattern['hierarchy'][[7,8,9,10]] = ['Left','CA1','Anterior']
stream_pattern['positions'][[7,8,9,10]] = [-28-CA1_OFFSET,-10,-22]

stream_pattern['hierarchy'][[]] = ['Left','CA1','Posterior']
stream_pattern['positions'][[]] = [-25-CA1_OFFSET,-37,-0]

stream_pattern = stream_pattern[metadata['all_streams']]

metadata['stream_pattern'] = stream_pattern