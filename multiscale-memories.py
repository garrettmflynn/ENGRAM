from engram.declarative import ID
from engram.episodic import shaders
from settings import ramconfig
import os

os.system('clear')

existingEngrams = False

# model_params = np.load('/models/OpenBCI_02_20_20.pkl')
settings = ramconfig.settings

if not existingEngrams:

    regions = ramconfig.regions

    id = ID(name='keck', extension='.ns3', project='RAM', settings=settings)
    id.loadTrace(method='name', regions=regions)
    id.loadEvents(extension='.nex')
    id.preprocess(settings=settings)
    id.save()

else:
    id = ID(name='keck').load()
    print('Loaded!')


id.model('channels', 'MD')

# from visbrain.gui import Signal
# from visbrain.utils import generate_eeg

# xlabel = 'Time (ms)'
# ylabel = 'Amplitude (uV)'
# title = 'Plot of a 1-d signal'

# Signal(id.traces['Trace0']['Data'][10][0:1000000], sf=id.traces['Trace0']['fs'], xlabel=xlabel, ylabel=ylabel,
#        title=title,form='tf',tf_cmap='rdbu').show()

# shaders.select('spectrogram',id.traces['Session0']['data'][0],settings)
