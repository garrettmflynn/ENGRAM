'''
This module defines :class:`Engram`, the container for all offline analysis.
It contains many :class:`Mneme` objects that are:

    - Labeled with their region of origin
    - Organized into trial & channel subsections.


Each :class:`Engram` has a unique event tag.
'''

import datetime
from engram import procedural
import numpy as np


class Engram(object):
    def __init__(self, engram, id='User', tag='Unspecified'):
        self.id = id
        self.tag = tag
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.trials = engram
        self.models = {}

    def __repr__(self):
        return "Engram('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)
