# -*- coding:utf-8 -*-
'''
:mod:'engram.declarative' provides classes for containing neurophysiology recordings and derivative features.

Classes from :mod:'engram.declarative' return nested data structures containing one or more classes from this module.

Classes:

.. autoclass:: ID

.. autoclass:: Duration

.. autoclass:: Bin

.. autoclass:: Cont

'''

import engram
from engram.declarative.id import ID
from engram.declarative.duration import Duration
from engram.declarative.bin import Bin
from engram.declarative.cont import Cont

objectlist = [ID, Duration, Bin, Cont]

objectnames = [ob.__name__ for ob in objectlist]
class_by_name = dict(zip(objectnames, objectlist))