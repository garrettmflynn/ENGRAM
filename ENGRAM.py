import engram
from engram.declarative import ID
import os

# Clear Terminal
clear = lambda: os.system('clear') # 'cls' for Windows | 'clear' for Linux
clear()

existingEngrams = True



if not existingEngrams:
    # Specify Channel Regions and Positions
    regions = {}
    regions['laCA3'] = {}
    regions['laCA1'] = {}
    regions['raCA3'] = {}
    regions['raCA1'] = {}
    regions['rpCA3'] = {}
    regions['rpCA1'] = {}
    regions['laCA3']['channels'] = {1,2,3,4,5,6}
    regions['laCA1']['channels'] = {7,8,9,10}
    regions['raCA3']['channels'] = {17,18,19,20,21,22}
    regions['raCA1']['channels'] = {23,24,25,26}
    regions['rpCA3']['channels'] = {33,34,35,36,37,38}
    regions['rpCA1']['channels'] = {39,40,41,42}
    regions['laCA3']['position'] = (-28,-10,-22)
    regions['laCA1']['position'] = (-28,-10,-22)
    regions['raCA3']['position'] = (28,-10,-22)
    regions['raCA1']['position'] = (28,-10,-22)
    regions['rpCA3']['position'] = (25,-37,-0) # Not accounting for CA1 vs CA3
    regions['rpCA1']['position'] = (25,-37,-0)

    #  (MNI coordinates: right hippocampus 28 -10 -22
    # (most anterior); 30 -14 -18; 30 -18 -16; 30 -22 -14; 30 -26 -12; 30 -29 -10; 30 -33 -6; 25 -37
    # 0 (most posterior), left hippocampus -28 -10 -22 (most anterior); -30 -14 -18; -30 -18 -16;
    # -30 -22 -14; -30 -26 -12; -30 -29 -10; -30 -33 -6; -25 -37 0 (most posterior). 

    # Specify Settings
    settings = {
        'fs':2000,
        'all_channels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        'feature':'STFT',
        'bandpass_min':1,
        'bandpass_max':250,
        '2D_min':0,
        '2D_max':150,
        't_bin':.1, # 100 ms is .1 for blackrock
        'f_bin':.5,
        'overlap': .05,
        'norm': True,
        'norm_method':'ZSCORE',
        'log_transform': True,
        'mneme_method':'roi',
        'roi_bounds': (-1,1) # Two-second window centered at the event
    }


    id = ID(name='keck',extension='.ns3',project='RAM',settings=settings)
    id.loadTrace(regions=regions)
    id.loadEvents(extension='.nex')
    id.createEngrams()
    id.save()

else:
    ## Load ID
    id = ID(name='keck').load()