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
* Tensorflow (**must install manually**)
* Numpy
* Scipy
* Vispy
* PyQt5

Features
-----------

* **Convert electrophysiology data from multiple brain regions into Engrams** using ``engram.declarative``
* **Model multi-channel electrophysiology recordings** using multiple machine learning techniques (i.e. MIMO, CNN, RNN, etc) using ``engram.procedural``
* **Visualize multi-input multi-output (MIMO) modeling** of electrophysiology recordings using ``engram.episodic``
* Leverage ROOTS_ to **grow artificial connections between functionally connected neurons**
* **Online data processing for OpenBCI headsets** using ``engram.working``