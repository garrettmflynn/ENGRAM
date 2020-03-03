# -*- coding:utf-8 -*-
'''
:mod:'engram.data' provides classes for storing neurophysiology recordings
and functions for preprocessing this data.

Classes from :mod:'engram.data' return nested data structures containing one or more classes from this module.

Classes:

.. autoclass:: ID

.. autoclass:: Engram

.. autoclass:: Trace

.. autoclass:: Mneme

'''

import engram
from engram.data.id import ID
from mneme.data.trace import Trace
from mneme.data.engram import Engram
from mneme.data.mneme import Mneme

from mneme.data.neo_handler import unpackNeo


objectlist = [ID,Trace,Engram,Mneme]

objectnames = [ob.__name__ for ob in objectlist]
class_by_name = dict(zip(objectnames,objectlist))