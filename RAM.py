from engram.declarative import ID
from settings import ramconfig
import os

# Clear Terminal
clear = lambda: os.system('clear') # 'cls' for Windows | 'clear' for Linux
clear()

existingEngrams = True

#model_params = np.load('/Users/garrettflynn/Desktop/MOUSAI/Mneme/models/OpenBCI_02_20_20.pkl')
settings = ramconfig.settings

if not existingEngrams:

    # Specify Channel Regions and Positions
    regions = ramconfig.regions

    id = ID(name='keck',extension='.ns3',project='RAM',settings=settings)
    id.loadTrace(method='name',regions=regions)
    id.loadEvents(extension='.nex')
    id.createEngrams()
    id.save()

else:
    ## Load ID
    id = ID(name='keck').load()
    print('Loaded!')


id.model('channels','CNN')
