ENGRAM
========

**EN**\coding **G**\raphical **R**\epresentations of **A**\ctivated **M**\emories is a
Python package for developing cognitive neural prostheses.

Organization and Philosophy
-----------------------------

ENGRAM includes four modules for cortical prosthesis development:

``engram.declarative`` contains classes for containing data.
  - Organize and standardize engrams

``engram.procedural`` contains code for managing pipeline processes.
  - Train multi-input multi-output (MIMO) models
  - Train information decoding models

``engram.episodic`` contains code used for graph(ics) generation.
  - Visualize model weights
  - Visualize functional connectivity

``engram.working`` contains code for online data processing.
  - Stream data from OpenBCI (or synthetic) boards

  See examples for usage conventions. 

Installation
------------

Install ENGRAM by running:

    install engram

Contribute
----------

- Issue Tracker: github.com/garrettmflynn/engram/issues
- Source Code: github.com/garrettmflynn/engram

Support
-------

If you are having issues, please email me at garrett@garrettflynn.com

More information
----------------

- Documentation: http://engram.readthedocs.io/

License
----------------
:license: GNU General Public License v3 (GPLv3), see LICENSE.txt for details.
