from engram.declarative import ID
from engram.procedural import events
from settings import configtemplate
from scipy.io import loadmat
import neo
import numpy as np
import os

os.system('clear')

existingEngrams = False

# model_params = np.load('/models/OpenBCI_02_20_20.pkl')
metadata = {}
metadata = ramconfig.metadata

if not existingEngrams:

    # Load Signals Using Neo
    tracedir = 'raw'
    filename = os.path.join(tracedir, f"{metadata['name']}",
                                            f"{metadata['name']}{metadata['extensions']['signals']}")
    reader = neo.get_io(filename=filename)

    blks = reader.read(lazy=False)
    for blk in blks:
        for seg in blk.segments:
            raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0)
            float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')
            data = np.transpose(float_sigs)
            fs = reader.get_signal_sampling_rate()
            units = reader.header['signal_channels'][0]['units']

    print('GET NEURONS FROM HERE INSTEAD')
    data = data[np.asarray(metadata['all_streams'])-1]

    # Load Events
    eventsname = os.path.join(tracedir, f"{metadata['name']}",
                                        f"{metadata['name']}{metadata['extensions']['events']}")
    reader = neo.get_io(filename=eventsname)
    events_, spikes_ = events.select(metadata['project'], reader)

    # Load Trial Labels
    labelsname = os.path.join(tracedir, f"{metadata['name']}",
                                            f"{metadata['name']}_labels.mat")
    
    trial_labels = {}
    labels = loadmat(labelsname)
    keys_list = list(labels)
    for key in keys_list:
        if 'Label' in key:
            name = key[6:]
            trial_labels[name] =  np.squeeze(labels[key])

    # Load Spikes (convert spikes to binary array + derive source channel)
    bin_chans = []
    spike_times = []
    for neuron in spikes_:
        spike_times.append(spikes_[neuron])
        bin_chans.append(int(neuron[3:6].lstrip('0')))

    id = ID(metadata)
    # NO LOCATIONS YET
    id.addDuration(conts=data, cont_channels=metadata['all_streams'], \
        bin_timestamps=spike_times, bin_channels=bin_chans, events=events_, \
        labels=trial_labels)
    
    for ii, _ in enumerate(id.durations):
        for jj, _ in enumerate(id.durations[ii].bins):
            if id.durations[ii].conts[jj]:
                id.durations[ii].bins[jj].makeVectorsFromTimestamps(np.size(id.durations[ii].conts[jj].data,1))
            else:
                id.durations[ii].bins[jj].makeVectorsFromTimestamps()
    id.save()

else:
    id = ID(metadata).load()
    print('Loaded!')

id.standardize(form='stft')
id.extractTrials()
id.episode(shader='engram')

# id.model('channels', 'MD')