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
    id.preprocess(settings=settings)
    id.save()

else:
    id = ID(name='keck').load()
    print('Loaded!')

id.episode(shader='engram')

# id.model('channels', 'MD')

# from ephyviewer import mkQApp, MainViewer, TraceViewer, TimeFreqViewer
# from ephyviewer import InMemoryAnalogSignalSource
# import ephyviewer
# import numpy as np

# #you must first create a main Qt application (for event loop)
# app = mkQApp()

# #create fake 16 signals with 100000 at 10kHz
# sigs = id.traces['Trace0']['Data'].T
# sample_rate = id.traces['Trace0']['fs']
# t_start = 0.

# #Create the main window that can contain several viewers
# win = MainViewer(debug=True, show_auto_scale=True)

# #Create a datasource for the viewer
# # here we use InMemoryAnalogSignalSource but
# # you can alose use your custum datasource by inheritance
# source = InMemoryAnalogSignalSource(sigs, sample_rate, t_start)

# #create a viewer for signal with TraceViewer
# view1 = TraceViewer(source=source, name='trace')
# view1.params['scale_mode'] = 'same_for_all'
# view1.auto_scale()

# #create a time freq viewer conencted to the same source
# view2 = TimeFreqViewer(source=source, name='tfr')

# view2.params['show_axis'] = False
# view2.params['timefreq', 'deltafreq'] = 1
# view2.by_channel_params['ch3', 'visible'] = True


# #add them to mainwindow
# win.add_view(view1)
# win.add_view(view2)


# #show main window and run Qapp
# win.show()
# app.exec_()

# from visbrain.gui import Signal
# from visbrain.utils import generate_eeg

# xlabel = 'Time (ms)'
# ylabel = 'Amplitude (uV)'
# title = 'Plot of a 1-d signal'

# Signal(id.traces['Trace0']['Data'][10][0:1000000], sf=id.traces['Trace0']['fs'], xlabel=xlabel, ylabel=ylabel,
#        title=title,form='tf',tf_cmap='rdbu').show()

# shaders.select('spectrogram',id.traces['Session0']['data'][0],settings)
