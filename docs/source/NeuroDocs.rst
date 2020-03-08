.. _neurodocs:

===============================================================
Notes on Documentating the Brain
===============================================================

.. important::

  **This page details the core developer's personal exploration**
  **of neuroscience software documentation.**
  
  For ENGRAM's documentation, see :ref:`getting-started` or our :ref:`api`.

The Problem
--------------------------------------------

**Neuroscience Documentation Sucks:** Yes, yes it does.

A Selection of Cases
--------------------------------------------
Wagner Lab
^^^^^^^^^^^^^^^^^^^^

**Entry Vignette** to provide the reader with an inviting Introduction
to the feel of the context in which the case takes place

`Wagner Lab`_ is a memory lab at Stanford University that releases all of their 
code with extensive documentation 
and enough functionality to reproduce publication results.
:cite:`Gagnon2018`
:cite:`Waskom2017`

**An Introduction** to familiarize the reader with the central features
including rationale and research procedures

**An Extensive Narrative Description** to of the case(s) and its context,
which may involve historical or organizational information important for understanding the case

.. important:: We have found **Theme #1**

**Draw from Additional Data Sources** and integrate with the researcher's own interpretations
of the issues and both confirming and disproving evidence are presented followed by the
presentation of the overall case assertions

**A Closing Vignette** as a way of cautoning the reader to the specific case context
saying "I like to close on an experiential note, reminding the reader that this report
is just one person's encounter with a complex case"


EEGLearn
^^^^^^^^^^^^^^^^^^^^

EEGLearn_ is a set of functions for supervised feature learning/classification 
of mental states from EEG based on "EEG images" idea. 
:cite:`Bashivan2016`

.. important:: We have found **Theme #2**

Ephyviewer
^^^^^^^^^^^^^^^^^^^^
Ephyviewer_ is a Python library based on pyqtgraph 
for building custom viewers for electrophysiological signals,
video, events, epochs, spike trains,
data tables, and time-frequency representations of signals.

.. important:: We have found **Theme #3**

Neurotic
^^^^^^^^^^^^^^^^^^^^

Neurotic_ is an app for Windows, macOS, and Linux that allows you to 
easily review and annotate your electrophysiology data and simultaneously 
captured video.

.. important:: We have found **Theme #4**

Neo
^^^^^^^^^^^^^^^^^^^^

Neo_ is a Python package for working with electrophysiology data in Python,
together with support for reading a wide range of neurophysiology file formats,
including Spike2, NeuroExplorer, AlphaOmega, Axon, Blackrock, Plexon, Tdt, 
and support for writing to a subset of these formats 
plus non-proprietary formats including HDF5. 
:cite:`Garcia2014`


.. important:: We have found **Theme #5**

Elephant
^^^^^^^^^^^^^^^^^^^^

Elephant_ (Electrophysiology Analysis Toolkit) is an 
emerging open-source, community centered library 
for the analysis of electrophysiological data 
in the Python programming language. 


.. important:: We have found **Theme #6**


Preliminary Patterns
----------------------------------------------------------

.. admonition:: Lesson #1
    
    *???*


Appendix: Some Cool Things You Can Do with Sphinx
-----------------------------------------------------

.. include:: diary/March2020/4.rst

.. jupyter-execute:: 

  from engram.episodic import shaders
  shaders.select('fireworks')


References
--------------------------------------------

.. bibliography:: references.bib

.. _Ephyviewer:     https://github.com/NeuralEnsemble/ephyviewer
.. _EEGLearn:       https://github.com/pbashivan
.. _Wagner Lab:     https://github.com/WagnerLabPapers
.. _Neurotic:       https://github.com/jpgill86/neurotic
.. _Elephant:       https://elephant.readthedocs.io/en/latest/
.. _Neo:            https://github.com/NeuralEnsemble/python-neo