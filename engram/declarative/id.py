""" 
This module defines :class:'ID'
"""

import datetime
import neo
from engram.data.neo_handler import unpackNeo

class ID(object):
    def __init__(self, name=None):
        """
        This is the constructor for the ID data object,
        which contains all Traces and Engrams of a given user tracked by ENGRAM
        """

        self.id = id
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.engrams = {}
        self.traces = {}
        self.settings = {}

        if not os.path.exists(datadir):
            os.mkdir(datadir)
        filename = os.path.join(datadir, f"{self.id}")
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)
        print(self.id + " initialized!")



    def __repr__(self):
        return "ID('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)

    def loadTrace(self,session="Trace" + str(len(self.trace))):
        print('Loading new trace...')
        reader = neo.get_io(filename=self.id)
        self.trace[session] = unpackNeo(reader)

    def makeEngrams(self):
        
        self.engrams = {}
        self.settings = self.settings 

        warn('In development. No engrams resulting.')

    def update(self,datadir='users'):
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        filename = os.path.join(datadir, f"{self.id}")
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)
        print(self.id + " updated!")