Core Modules
=============

This library includes several modules for cortical prosthesis development:

``engram.declarative`` contains classes for containing data.
  - Load neural recordings into our nested data structures (Data —> Mnemes —> Engrams —> IDs)

``engram.procedural`` contains code for managing pipeline processes.
  - Train multi-input multi-output (MIMO) models
  - Train information decoding models

``engram.episodic`` (high-level **S**\hader **O**\ptimized **D**\ata **I**\nteraction **C**\ommands)
contains code used for graph visualization and other graphical strategies.
  - Visualize model weights

These modules should serve as a solid foundation for your own domain-specific prostheses.
