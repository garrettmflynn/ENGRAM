
ENGRAM
========

**EN**\coding **G**\raphical **R**\epresentations of **A**\ctivated **M**\emories is a
Python package for cognitive neural prostheses.

Organization and Philosophy
-----------------------------

This library includes several modules for cortical prosthesis development:

``engram.data`` contains code for holding and processing data.
  - Load neural recordings into our nested data structures (Data —> Mnemes —> Engrams —> IDs)
  - Preprocess data before encoding
``engram.encode`` contains code for managing pipeline processes.
  - Train multi-input multi-output (MIMO) models
  - Train information decoding models
``engram.represent`` contains code used for visualization.
  - Visualize model weights

Though packaged as a fully-functioning application, these ENGRAM modules should serve 
as a solid foundation for your own domain-specific prostheses.

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

If you are having issues, please email me at gflynn@usc.edu

More information
----------------

- Documentation: http://engram.readthedocs.io/

License
----------------
:license: GNU General Public License v3 (GPLv3), see LICENSE.txt for details.
