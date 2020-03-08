.. _getting-started:

================
Getting Started
================

**Encoding Graphical Representations of Activated Memories (ENGRAM)**
is an open-source Python package for developing cognitive neural prostheses.

Installation
-------------
Get engram from pip:

``pip install engram``


Requirements
-------------

* **Python 3.7**
* Neo
* Tensorflow
* Numpy
* Scipy
* Pandas
* Glumpy

.. important:: 
  To use Glumpy, you must install ``pyopengl`` and ``freetype-py`` manually, then change the ``glumpy/ext/__init__.py`` to install this local version of ``freetype`` rather than the included Glumpy version.

Features
-----------

* **Convert electrophysiology data from multiple brain regions into Engrams** using ``engram.declarative``
* **Model multi-channel electrophysiology recordings** using multiple machine learning techniques (i.e. MIMO, CNN, RNN, etc) using ``engram.procedural``
* **Visualize multi-input multi-output (MIMO) modeling** of electrophysiology recordings using ``engram.episodic``
* Leverage ROOTS_ to **grow artificial connections between functionally connected neurons**
* **Online data processing for OpenBCI headsets** using ``engram.working``

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

.. _ROOTS:          https://github.com/bingsome/roots