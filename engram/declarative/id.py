""" 
This module defines :class:'ID'
"""
import os
import datetime
import neo
import pickle
from engram.procedural.neo_handler import unpackNeo
from engram.declarative.engram import Engram
from engram.declarative.mneme import Mneme
from engram.procedural import event_parsers,data_parser,feature_parser
import numpy as np

class ID(object):
    def __init__(self, name=None,extension='.ns3',project='RAM'):
        """
        This is the constructor for the ID data object,
        which contains all Traces and Engrams of a given user tracked by ENGRAM
        """

        self.id = name
        self.project=project
        self.extension = extension
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.engrams = {}
        self.traces = {}
        self.settings = {}
        self.details = {}
        self.regions = []

        datadir = 'users'
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

    def loadTrace(self,session=None,regions=None):
        if session is None:
            session = "Trace" + str(len(self.traces))
        self.traces[session] = []
        self.details[session] = {}

        print('Loading new trace...')
        tracedir = 'raw'
        filename = os.path.join(tracedir, f"{self.id}",f"{self.id}{self.extension}")
        reader = neo.get_io(filename=filename)
        data,fs,units = unpackNeo(reader)
        self.traces[session] = data
        self.details[session]['fs'] = fs
        self.details[session]['units'] = units

        if regions is not None:
            self.details[session]['regions'] = regions
            if self.regions is None:
                self.regions = np.empty()
            for region in regions:
                np.append(self.regions,region)
            self.regions = np.unique(self.regions)

    def loadEvents(self,session=None,extension='.nex'):
        if session is None:
            session = "Trace" + str(len(self.traces)-1)
        tracedir = 'raw'
        filename = os.path.join(tracedir, f"{self.id}",f"{self.id}{extension}")
        reader = neo.get_io(filename=filename)
        self.details[session]['events'],self.details[session]['neurons'] = event_parsers.select(self.project,reader)


    def createMnemes(self,settings=None):
            # specify self.settings
            self.mnemes = {}

            # Create Mnemes from Traces
            for trace in self.traces:
                for event in self.details[trace]['events']:
                    times = self.details[trace]['events'][event]
                    self.mnemes[event] = np.empty(len(times))
                    for time in times
                        data = data_parser.getData(self.traces[trace],time,self.details[session]['fs'])
                        feature = feature_parser.select(data,settings)
                        self.mnemes[event] = Mneme(self.id,event,feature)

    def createEngrams(self):
        # specify self.settings
        self.engrams = {}

        # Create Engrams from Mnemes
        for region in self.regions:
            self.engrams[region] = Engram(self.id,)


    def update(self,datadir='users'):
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        filename = os.path.join(datadir, f"{self.id}.ns3")
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)
        print(self.id + " updated!")