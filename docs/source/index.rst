.. module:: engram
.. module:: engram.declarative
.. module:: engram.procedural
.. module:: engram.episodic



==================
ENGRAM
==================

**Encoding Graphical Representations of Activated Memories (ENGRAM)**
is an open source Python package 
for developing cognitive neural prostheses.

|PyPI badge| |GitHub badge| |Docs badge| |Travis badge| |License badge|

Key Features
---------------
* **Convert electrophysiology data from multiple brain regions into Engrams** using ``engram.declarative``
* **Model multi-channel electrophysiology recordings** using multiple machine learning techniques (i.e. MIMO, CNN, RNN, etc) using ``engram.procedural``
* **Visualize multi-input multi-output (MIMO) modeling** of electrophysiology recordings using ``engram.episodic``
* Grow **artificial connections between functionally connected neurons**
* **Online data processing for OpenBCI headsets** using ``engram.working``

.. toctree::
    :maxdepth: 2
    :hidden:
    
    GettingStarted
    .. Walkthroughs
    .. FurtherReadings
    API
    Contributing
    ReleaseNotes
    Acknowledgements

.. |PyPI badge| image:: https://img.shields.io/pypi/v/engram.svg?logo=python&logoColor=white
    :target: PyPI_
    :alt: PyPI project

.. |GitHub badge| image:: https://img.shields.io/badge/github-source_code-blue.svg?logo=github&logoColor=white
    :target: GitHub_
    :alt: GitHub source code

.. |Docs badge| image:: https://img.shields.io/readthedocs/engram/latest.svg?logo=read-the-docs&logoColor=white
    :target: ReadTheDocs_
    :alt: Documentation status

.. |License badge| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: License_
    :alt: License

.. |Travis badge| image:: https://img.shields.io/travis/com/garrettmflynn/engram/master.svg?logo=travis-ci&logoColor=white
   :target: Travis_
   :alt: Travis build status

.. _GitHub:         https://github.com/garrettmflynn/engram
.. _ReadTheDocs:    https://readthedocs.org/projects/engram
.. _PyPI:           https://pypi.org/project/engram/
.. _License:        https://www.gnu.org/licenses/gpl-3.0
.. _Travis:         https://travis-ci.com/github/GarrettMFlynn/ENGRAM