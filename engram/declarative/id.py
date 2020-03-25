'''
This module defines :class:`ID`, the main container gathering all the data,
whether discrete or continous, for a given recording session.
It is the container for the :class:`Engram` class.
'''
import os
import datetime
import neo
import pickle
from engram.procedural.neo_handler import unpackNeo
from engram.declarative.engram import Engram
from engram.declarative.mneme import Mneme
from engram.procedural import events, data, features, filters, train
from engram.episodic import shaders
import numpy as np
from scipy.io import loadmat


class ID(object):

    '''
    Main container gathering all the data, whether discrete or continous, for a
    given recording session.
    '''

    def __init__(self, name=None, extension=None, project=None,
                 settings=None, load=False):

        self.id = name
        self.project = project
        self.extension = extension
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.trial_features = []
        self.trial_labels = []
        self.traces = {}
        self.settings = settings
        self.regions = []

    def __repr__(self):
        return "ID('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)

    def loadTrace(self, method='name', session=None,
                  manual=None, regions=None):

        if session is None:
            session = "Trace" + str(len(self.traces))
        self.traces[session] = {'Data': [], 'fs': None,
                                'units': None, 'regions': {}, 'events':{},'spikes':[],'labels':{}}

        print('Loading new trace...')

        if method == 'name':
            tracedir = 'raw'
            filename = os.path.join(tracedir, f"{self.id}",
                                              f"{self.id}{self.extension}")
            reader = neo.get_io(filename=filename)
            data, fs, units = unpackNeo(reader)

        elif method == 'manual':
            print('Loading channel data manually')
            data = manual[0]
            fs = manual[1]
            units = manual[2]

        # Get specified channels from data
        data = data[np.asarray(self.settings['all_channels'])-1]

        # Only downsample
        if fs != self.settings['fs'] and self.settings['fs'] < fs:
            data = filters.select('bandpass', min=0, max=self.settings['fs'],
                                  fs=fs, order=5)
            downsample = round(fs/self.settings['fs'])
            self.traces[session]['fs'] = fs/downsample
            data = data[0::downsample]
            print('Downsampled to ' + self.traces[session] + 'Hz')
        else:
            self.traces[session]['fs'] = fs
        self.traces[session]['Data'] = data
        self.traces[session]['units'] = units

        if regions is not None:
            self.traces[session]['regions'] = regions
            if self.regions is None:
                self.regions = np.empty()
            for region in regions:
                self.regions = np.append(self.regions, region)
            self.regions = np.unique(self.regions)

    def loadEvents(self, session=None, extension='.nex'):
        if session is None:
            session = "Trace" + str(len(self.traces)-1)

        # add events and spikes
        tracedir = 'raw'
        eventsname = os.path.join(tracedir, f"{self.id}",
                                          f"{self.id}{extension}")
        reader = neo.get_io(filename=eventsname)
        self.traces[session]['events'], spikes_ = events.select(self.project, reader)

        # add labels
        labelsname = os.path.join(tracedir, f"{self.id}",
                                              f"{self.id}_labels.mat")
        labels = loadmat(labelsname)
        keys_list = list(labels)
        for key in keys_list:
            if 'Label' in key:
                name = key[6:]
                self.traces[session]['labels'][name] =  np.squeeze(labels[key])

        # convert spikes to binary array + derive source channel
        self.settings['spike_channels'] = []
        for neuron in spikes_:
            spikes = np.zeros(np.size(self.traces[session]['Data'],1))
            rounded_indices = np.round(spikes_[neuron]*self.traces[session]['fs']).astype('int')
            spikes[rounded_indices] = 1

            self.traces[session]['spikes'].append(spikes)
            self.settings['spike_channels'].append(int(neuron[3:6].lstrip('0')))

        self.traces[session]['spikes'] = np.array(self.traces[session]['spikes']).T

    def preprocess(self, settings=None):

        # trials x sources x time x etc 
        # note: all sources need their true corresponding address (for region specification)
        trial_matrix = []
        label_matrix = []

        for trace in self.traces:

            # Derive Features from Each Trace
            feature, self.settings['t_feat'], self.settings['f_feat'] = features.select(
                                            self.settings['feature'],
                                            self.traces[trace],
                                            self.settings
                                            )

            times = self.traces[trace]['events'][self.settings['event_of_interest']]

            for trial,time in enumerate(times):
                # Select Proper Timebins from Features
                if 'prev_len' in locals():
                    featureset, prev_len = data.select(feature=feature,
                                                        time=time, settings=self.settings,
                                                        prev_len=prev_len)
                else:
                    featureset, prev_len = data.select(feature=feature,
                                                        time=time, settings=self.settings)
                trial_matrix.append(featureset)
                print('Trial ' + str(trial) + ' finished.')

            if not self.trial_features:
                self.trial_features = trial_matrix
                self.trial_labels = self.traces[trace]['labels']
            else:
                self.trial_features.append(trial_matrix)
                self.trial_labels.append(self.traces[trace]['labels'])
        print('Engrams completed!')

    def model(self, method='channels', model_type='CNN'):
        train.train(model_type, self.trial_features, self.trial_labels)

    def save(self, datadir='users'):
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        filename = os.path.join(datadir, f"{self.id}")
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)
        print(self.id + " saved!")

    def load(self, datadir='users'):
        filename = os.path.join(datadir, f"{self.id}")
        loadedID = pickle.load(open(filename, "rb"))
        print(loadedID.id + " loaded!")

        return loadedID

    def episode(self, shader='engram'):
        regions = self.traces['Trace0']['regions']
        data = self.traces['Trace0']['spikes']
        assignments = self.settings['spike_channels']
        shaders.select(shader=shader,regions=regions,
                        data=data,assignments=assignments)
