import engram
from engram.declarative import ID

id = ID(name='keck',extension='.ns3',project='RAM')

# Specify Channel Regions
regions = {}
regions['CA3'] = (1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 22)
regions['CA1'] = (7, 8, 9, 10, 23, 24, 25, 26)

# Load Trace
id.loadTrace(regions=regions)
id.loadEvents(extension='.nex')

# Specify 
settings = {'feature':'STFT',
'bandpass_min':1,
'bandpass_max':250,
'2D_min':0,
'2D_max':150,
't_bin':.1, # 100 ms is .1 for blackrock
'f_bin':.5,
'overlap': .05
}

id.createMnemes(settings=settings)
id.createEngrams()
