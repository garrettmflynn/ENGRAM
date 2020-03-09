from engram.declarative import ID
from engram.episodic import shaders
from settings import ramconfig
import os

os.system('clear')

existingEngrams = True

# model_params = np.load('/models/OpenBCI_02_20_20.pkl')
settings = ramconfig.settings

if not existingEngrams:

    regions = ramconfig.regions

    id = ID(name='keck', extension='.ns3', project='RAM', settings=settings)
    id.loadTrace(method='name', regions=regions)
    id.loadEvents(extension='.nex')
    id.createEngrams()
    id.save()

else:
    id = ID(name='keck').load()
    print('Loaded!')


# id.model('channels', 'CNN')

shaders.select('spectrogram',id.traces['Session0']['data'][0],settings)
