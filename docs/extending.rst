Extending
*********

Overview of Pychell
===================

pychell is a fully objected oriented package, so most new functionality is done through defining new Python classes. A very powerful way of modifying the current state of pychell for supported spectrographs is to "patch" classes through the fastcore package.

Define A New Spectrograph
=========================

Each implemented spectrograph must live in a file named spectrograph_name.py (must be lowercase) in the data module. This file must define a new class called {INSTNAME}Parser which extends the base Parser class.