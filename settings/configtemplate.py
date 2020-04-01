#! /usr/bin/env python

settings = {
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

regions = {
        'Right': {
            'streams': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'position': [-100,0,0]
        },

        'Left': {
            'streams': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            'position': [100,0,0]
        },

        'Center': {
            'streams': [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            'position': [0,0,0]
        },
    }