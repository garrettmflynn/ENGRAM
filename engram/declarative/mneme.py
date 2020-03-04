""" 
This module defines :class:'Mneme'
"""

from engram.procedural.features import *

import datetime

class Mneme(object):
    def __init__(self, id,tag,feature):
        """
        This is the constructor for the Mneme data object,
        the smallest unit of memory tracked by ENGRAM.
        """

        self.id = id
        self.date = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.tag = tag
        self.feature = feature


    def __repr__(self):
        return "Mneme('{},'{}',{})".format(self.id, self.date)

    def __str__(self):
        return '{} _ {}'.format(self.id, self.date)