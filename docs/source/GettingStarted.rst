================
Getting Started
================

Encoding Graphical Representations of Activated Memories (ENGRAM) 
is an open-source Python package for developing cognitive neural prostheses.

Installation
-------------
Get engram from pip:

``pip install engram``


Requirements
-------------
**Python 3.7**
Neo
Brainflow
Tensorflow
Glumpy

.. note:: Must install ``pyopengl`` and ``freetype-py`` manually, then change the ``glumpy/ext/__init__.py`` to install this local version of ``freetype`` rather than the included Glumpy version.

Numpy
Scipy
Pandas

Core Modules
-------------

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

These modules should serve as a solid foundation for your own domain-specific prostheses.


Examples
---------


multiscale-memories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Song Lab* (2020)

`multiscale-memories <https://github.com/GarrettMFlynn/multiscale-memories>`_
is a custom pipeline developed at Song Lab 
for decoding memory contents from human hippocampal recordings.

kinesis-v2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Ahmanson Lab* (2020)

`kinesis-v2 <https://github.com/Mousai-Neurotechnologies/kinesis-v2>`_
is a movement decoding pipeline for OpenBCI headsets 
that integrates automatic motion tracking with real-time signals processing. 



episodic-memories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Garrett Flynn* (2020)

A personal project for shader-based representation of memories

.. jupyter-execute:: 

  name = 'world'
  print('hello ' + name + '!')