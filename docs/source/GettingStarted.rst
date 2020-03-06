Getting Started
================

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


Restoring Active Memory (RAM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Song Lab* (2020)

`multiscale-memories <https://github.com/GarrettMFlynn/multiscale-memories>`_
is a custom pipeline for decoding memory content from human hippocampal recordings.


Neurogenesis Working Group 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Ahmanson Lab* (2020)

`kinesis v2 <https://github.com/Mousai-Neurotechnologies/kinesis-v2>`_
is a movement decoding pipeline for OpenBCI headsets 
that integrates automatic motion tracking with real-time signals processing. 