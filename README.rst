
ENGRAM
========

*Encoding Graphical Representations of Activated Memories (ENGRAM) is
Python package for the design of cognitive neural prostheses.*

Organization and Philosophy
-----------------------------

This library includes several modules for cortical prosthesis development. These are separated by the level 
of the prosthesis that they would operate on (e.g. data processing vs UI).

- engram.data contains code for holding and processing data.
    * Train multi-input multi-output (MIMO) models
    * Train information decoding models
- engram.managers contains code for managing pipeline processes.
- engram.ui contains code used for visualization.
    * Visualize model weights

Though packaged as a fully-functioning application, ENGRAM should serve as a good foundation for your own 
domain-specific prostheses.

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
