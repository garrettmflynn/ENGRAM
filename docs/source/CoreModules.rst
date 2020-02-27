Core Modules
=============

This library includes several modules for cortical prosthesis development:

``engram.data`` contains code for holding and processing data.
  - Load neural recordings into our nested data structures (Data —> Mnemes —> Engrams —> IDs)
  - Preprocess data before encoding
``engram.encode`` contains code for managing pipeline processes.
  - Train multi-input multi-output (MIMO) models
  - Train information decoding models
``engram.graph`` contains code used for graph visualization.
  - Visualize model weights

These modules should serve as a solid foundation for your own domain-specific prostheses.
