'''
This module defines :class:`Duration`,  a container for unique periods of time.
'''
from engram.declarative.bin import Bin
from engram.declarative.cont import Cont

import numpy as np

class Duration(object):
    def __init__(self, id, bins=[], bin_channels = [], bin_timestamps=[],conts=[], cont_channels = [],events={},labels={},metadata=None):

        self.id = id

        if bins is not None:
            self.bins = [Bin(id, data=bins, timestamps=bin_timestamps, channel_labels=bin_channels, metadata=metadata)]
        else:
            self.bins = []
        if conts is not None:
            self.conts = [Cont(id, data=conts, channel_labels=cont_channels, metadata=metadata)]
        else:
            self.conts = []
        
        self.events = events
        self.trial_labels = np.asarray(labels)
        self.trials = None
        self.metadata = metadata

    def __repr__(self):
        return "Cont('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)

    # Data Manipulation
    def addBin(self, data = [], timestamps = [], channel_labels = []):
        self.bins.append(Bin(self.id,data=data, timestamps=timestamps, channel_labels=channel_labels, metadata=self.metadata))

    def addCont(self, data = [], timestamps = [], channel_labels = []):
        self.conts.append(Cont(self.id, data=data, timestamps=timestamps, channel_labels=channel_labels, metadata=self.metadata))

    def makeROIs(self):
        times = self.events[self.metadata['event_of_interest']]

        # Only continue if Conts and Bins have the same time vectors
        nconts = len(self.conts)
        nbins = len(self.bins)
        obj_names = ['Conts','Bins']
        objects = {obj_names[0]:self.conts, obj_names[1]: self.bins}
        trials = np.asarray([{obj_names[0]:[], obj_names[1]: []} for _ in range(len(times))])

        prev_len = None

        for trial,time in enumerate(times):
            for ii, obj_type in enumerate(objects):
                for jj, obj in enumerate(objects[obj_type]):
                    bounds = self.metadata['roi_bounds']
                    upper_index = (np.abs(obj.nD_labels['2D'] - (time + bounds[1]))).argmin()
                    lower_index = (np.abs(obj.nD_labels['2D'] - (time + bounds[0]))).argmin()
                    diff_one = upper_index-lower_index
                    if prev_len and prev_len != diff_one:
                        diff_two = prev_len - diff_one
                        if diff_one % 2:
                            center = lower_index+diff_one/2
                            offset = (prev_len+1)/2
                            upper_index = center + offset
                            lower_index = center - offset
                        upper_index += diff_two
                    if np.ndim(obj.data) == 2:
                        data = obj.data[:,lower_index:upper_index]
                    elif np.ndim(obj.data) == 3:
                        data = obj.data[:,lower_index:upper_index,:]
                    trials[trial][obj_type].append(self.container_factory(style=obj_type, data=data))
                    trials[trial][obj_type][-1].nD_labels['1D'] = obj.nD_labels['1D']
                    trials[trial][obj_type][-1].nD_labels['2D'] = obj.nD_labels['2D'][lower_index:upper_index]

                    prev_len = upper_index-lower_index

        self.trials = trials


    def container_factory(self,style=None, data = [], timestamps = [], channel_labels = []):
        if style == "Bins":
            return Bin(self.id, data=data, timestamps=timestamps, channel_labels=channel_labels, metadata=self.metadata)
        if style == "Conts":
            return Cont(self.id, data=data, timestamps=timestamps, channel_labels=channel_labels, metadata=self.metadata)
        else:
            raise Exception("Unrecognized container style.")