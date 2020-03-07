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
import numpy as np


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
        self.engrams = {}
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
                                'units': None, 'regions': {}}

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
            self.settings['fs'] = fs/downsample
            data = data[0::downsample]
            print('Downsampled to ' + self.settings['fs'] + 'Hz')
        self.traces[session]['Data'] = data
        self.traces[session]['units'] = units

        if regions is not None:
            self.traces[session]['regions'] = regions
            if self.regions is None:
                self.regions = np.empty()
            for region in regions:
                np.append(self.regions, region)
            self.regions = np.unique(self.regions)

    def loadEvents(self, session=None, extension='.nex'):
        if session is None:
            session = "Trace" + str(len(self.traces)-1)
        tracedir = 'raw'
        filename = os.path.join(tracedir, f"{self.id}",
                                          f"{self.id}{extension}")
        reader = neo.get_io(filename=filename)
        self.traces[session]['events'],
        self.traces[session]['neurons'] = events.select(self.project, reader)

    def createEngrams(self, settings=None):

        self.engrams = {}
        self.mnemes = {}

        for trace in self.traces:

            # Derive Features from Each Trace
            feature, self.settings['t_feat'],
            self.settings['f_feat'] = features.select(
                                            self.settings['feature'],
                                            self.traces[trace]['Data'],
                                            self.settings
                                            )

            for event in self.traces[trace]['events']:
                if event is not None and event != 'DIO_CHANGED':
                    times = self.traces[trace]['events'][event]
                    # engram = np.empty(len(times))
                    engram = {}

                    for trial, time in enumerate(times):
                        engram[trial] = {}

                        for channel in range(len(self.traces[trace]['Data'])):

                            # Select Proper Timebins from Features
                            if 'prev_len' in locals():
                                featureset,
                                prev_len = data.select(feature=feature[channel],
                                            time=time, settings=self.settings,
                                            prev_len=prev_len)
                            else:
                                featureset, 
                                prev_len = data.select(feature=feature[channel],
                                           time=time, settings=self.settings)

                            # Check Region of Origin
                            for region in self.traces[trace]['regions']:
                                if (self.settings['all_channels'][channel] 
                                in self.traces[trace]
                                ['regions'][region]['channels']):

                                    current_region = region

                            channel_name = self.settings['all_channels']
                            [channel]

                            engram[trial][channel_name] = {}
                            engram[trial]       \
                                [channel_name]  \
                                ['features'] = featureset

                            engram[trial][channel_name] \
                                ['region'] = current_region

                    if event not in self.engrams:
                        self.engrams[event] = None

                    self.engrams[event] = Engram(engram, id=self.id, tag=event)

            print('Engrams completed!')

    def model(self, method='channels', model_type='CNN'):
        in_matrix = []
        labels = []
        for engram in self.engrams:
            labels.append(self.engrams[engram].tag)
            tri_matrix = []
            if method == 'channels':
                for trial in self.engrams[engram].trials:
                    chan_matrix = []
                    for channel in self.engrams[engram].trials[trial]:

                        chan_matrix.append(
                            self.engrams[engram].trials[trial]
                            [channel]['features']
                            )

                    tri_matrix.append(chan_matrix)
                in_matrix.append(tri_matrix)
            elif method == 'regions':
                print('Method currently in development.')
            else:
                print('Method not supported.')

        train.train(model_type, in_matrix, labels)

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
