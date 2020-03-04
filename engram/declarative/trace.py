""" 
This module defines :class:'Trace'
"""

import sys
import imutils
import numpy as np
import time
import os
import neo
import pickle
import datetime

class Trace(object):
    def __init__(self, id='Default',tag=None):
        """
        This is the constructor for the Trace data object,
        from which all other ENGRAM classes will be derives
        """

        self.id = id
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.reader = []
        self.data = []
        self.details = {}

    def __repr__(self):
        return "Trace('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)