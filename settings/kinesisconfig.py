settings = {
        'fs': 100,
        'all_channels': range(16),
        'feature': 'STFT',
        'bandpass_min': 1,
        'bandpass_max': 250,
        '2D_min': 0,
        '2D_max': 150,
        't_bin': .1,  # 100 ms is .1 for blackrock
        'f_bin': .5,
        'overlap': .05,
        'norm': True,
        'norm_method': 'ZSCORE',
        'log_transform': True,
        'roi': 'events',
        'roi_bounds': (-1, 1),  # Two-second window centered at the event
        'model':  "/models/kinesis"
    }

