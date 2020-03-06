""" 
This module defines :class:'Engram'
"""

import datetime
from engram import procedural
import numpy as np

class Engram(object):
    def __init__(self, engram,id='User',tag='Unspecified'):
        """
        This is the constructor for the Engram data object,
        which contains the global pattern of activity evoked by a given event
        """

        self.id = id
        self.tag = tag
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.trials = engram
        self.models = {}

    def __repr__(self):
        return "Engram('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)