'''
This module defines :class:`Bin`,  a container for binary data.
'''

import numpy as np

class Bin(object):
    def __init__(self, id, data=[], timestamps = [], \
                channel_labels = [], settings=None):

        # Get number of elements in data
        numel = 1
        for dim in np.shape(data): numel *= dim

        # Check if data is continuous or binary
        if not ((data==0) | (data==1)).all() and not timestamps:
            print( "Invalid binary input. Nothing has been stored." )
        else:
            self.id = id
            self.timestamps = np.asarray(timestamps)
            self.data = np.asarray(data) # Channels x Time
            self.representation = 'raw'
            self.settings = settings

            self.nD_labels = {}
            self.nD_labels['1D'] = np.asarray(channel_labels)

            if np.ndim(data) <= 1:
                length = 0 # Time
                for source in self.timestamps:
                    length = np.ceil(np.maximum(np.max(source),length))
                self.nD_labels['2D'] = np.arange(0,length*self.settings['fs'])/self.settings['fs']
            else: 
                self.nD_labels['2D'] = np.arange(0,np.size(self.data, 1))/self.settings['fs']
            self.nD_labels['3D'] = None # ???

    def __repr__(self):
        return "Bin('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)

    def makeVectorsFromTimestamps(self,length=None):
        
        if not length:
            length =  0
            for source,idx in enumerate(self.timestamps):
                length = np.ceil(np.maximum(np.max(source),length))
            length = length * self.settings['fs']
        
        data = np.zeros((len(self.timestamps),length))
        for idx, times in enumerate(self.timestamps):
            rounded_indices = np.round(times*self.settings['fs']).astype('int')
            data[idx][rounded_indices] = 1

        self.data = data

    def bspline(self):
        print( 'TO DO' )
    
