# -*- coding:utf-8 -*-
"""
An open-source Python package for developing cognitive neural prostheses.
"""

import logging
logging_handler = logging.StreamHandler()

from .declarative import *
from .procedural import *
from .episodic import *

from .version import version as __version__