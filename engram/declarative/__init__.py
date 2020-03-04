# -*- coding:utf-8 -*-
'''
:mod:'engram.declarative' provides classes for storing neurophysiology recordings.

Classes from :mod:'engram.declarative' return nested data structures containing one or more classes from this module.

Classes:

.. autoclass:: ID

.. autoclass:: Engram

.. autoclass:: Trace

.. autoclass:: Mneme

'''

import engram
from engram.declarative.id import ID
from mneme.declarative.trace import Trace
from mneme.declarative.engram import Engram
from mneme.declarative.mneme import Mneme


objectlist = [ID,Trace,Engram,Mneme]

objectnames = [ob.__name__ for ob in objectlist]
class_by_name = dict(zip(objectnames,objectlist))