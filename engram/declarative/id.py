'''
This module defines :class:`ID`, the main container gathering all the data,
whether discrete or continous, for a given recording session.
It is the container for the :class:`Duration` class.
'''
import os
import datetime
import neo
import pickle
from engram.declarative.duration import Duration
from engram.procedural import events, filters, train
from engram.episodic import envs
import numpy as np
from scipy.io import loadmat


class ID(object):

    '''
    Main container gathering all the data, whether discrete or continous, for a
    given recording session.
    '''

    def __init__(self, metadata=None, load=False):

        self.id = metadata['name']
        self.project = metadata['project']
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.durations = []
        self.metadata = metadata

    def __repr__(self):
        return "ID('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)

    def addDuration(self, bins=[], bin_timestamps = [], bin_channels = [], conts=[], cont_channels = [], events={},labels={}):
        self.durations.append(Duration(self.id, bins, bin_channels, bin_timestamps, conts, cont_channels, events, labels, self.metadata))
   
    def model(self, method='channels', model_type='CNN'):
        train.train(model_type, self.trial_features, self.trial_labels)

    def save(self, datadir='users'):
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        filename = os.path.join(datadir, f"{self.id}")
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)
        print(self.id + " saved!")

    def load(self, metadata=None, datadir='users'):
        filename = os.path.join(datadir, f"{self.metadata['name']}")
        loadedID = pickle.load(open(filename, "rb"))
        print(loadedID.id + " loaded!")

        return loadedID

    def standardize(self,form='stft'):
        for ii,duration in enumerate(self.durations):

            # Derive Features from Each Trace
            for jj, cont in enumerate(duration.conts):

                if 'stft' in form:
                    self.durations[ii].conts[jj] = cont.stft()
                
                if 'lfp' in form:
                    self.durations[ii].conts[jj] = cont.stft()

            for jj, binary in enumerate(duration.bins):
                if 'bspline' in form:
                    self.durations[ii].conts[jj] = cont.stft()

    
    
    def extractTrials(self):
        for ii,duration in enumerate(self.durations):
            duration.makeROIs()

    def episode(self, shader='engram'):
        envs.select(shader=shader,id=self)
