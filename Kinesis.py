from engram.declarative import ID
from settings import kinesisconfig
from engram.working import streams,loggers
import os

# Clear Terminal
clear = lambda: os.system('clear') # 'cls' for Windows | 'clear' for Linux
clear()

stream = 'SYNTHETIC'
events =  ['FLOW']
port = None

settings = kinesisconfig.settings
regions = kinesisconfig.regions

manager = streams.DataManager(source=stream,event_sources=events,port=port)
keys = loggers.KeyLogger()
while True:
    manager.events.update()
    #categories = manager.events.sources[events[0]].categories
    #manager.predict(categories=categories,settings=settings)

    print(keys.pull())
    if keys.pull() == 'q':
        break

evs,evs_t = manager.event_sources.pull()
data = manager.pull()
data = data[manager.board.eeg_channels]
c_t = manager.board.time_channel
t = (data[c_t] - data[c_t][0])
session_length = manager.stop()

# Align events and brain data
manager.align()


# Specify real EEG positions
regions = {}
for idx,vals in enumerate(data):
    regions[str(idx)]['channels'] = idx
    regions[str(idx)]['positions'] = (idx/len(data),idx/len(data),idx/len(data))
fs = data.shape[1]/session_length
units = 'uV'

id = ID('neurogenesis')
id.loadTrace(method='manual', manual = (data,fs,units), regions=regions)

self.save(self.data, 'traces')