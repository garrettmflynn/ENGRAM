# -*- coding:utf-8 -*-
'''
ENGRAM is a package representing memory traces in the brain.
'''

import logging
logging_handler = logging.StreamHandler()

from engram.data import *
from engram.encode import *
from engram.episodic import *

from engram.version import version as __version__