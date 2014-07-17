Online Extreme Evolutionary Learning Machines

Code written by Joshua E. Auerbach

http://people.epfl.ch/joshua.auerbach

joshua.auerbach@epfl.ch

This code was used for the basis of the experiments reported in 

Auerbach, Joshua E., Chrisantha Fernando, and Dario Floreano.
Online Extreme Evolutionary Learning Machines.
14th International Conference on the Synthesis and Simulation of 
Living Systems (ALife XIV). New York, NY, July, 2014.

Full Text Available: https://mitpress.mit.edu/sites/default/files/titles/content/alife14/ch076.html


Documentation for this package is included in this README file.  

-------------
1. LICENSE
-------------

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License version 3 as published
by the Free Software Foundation (LGPL may be granted upon request). This 
program is distributed in the hope that it will be useful, but without any 
warranty; without even the implied warranty of merchantability or fitness for 
a particular purpose. See the GNU General Public License for more details.

---------------------
2. USAGE and SUPPORT
---------------------

This software is being release in the hopes that it will be useful for others
attempting to investigate the use of OEELMs.

The software is provided as is; however, we will do our best to maintain it 
and accommodate suggestions. If you want to be notified of future releases of 
the software or have questions, comments, bug reports or suggestions, send
an email to joshua.auerbach@epfl.ch

To run an experiment, do

 python oeelm.py RANDOM_SEED PARAMS_FILE_NAME OUTPUT_FILE_NAME

e.g.

 python oeelm.py 1 params/params_evolution_100_features evolution_100_features_seed_1.dat


---------------------
3. Dependencies
---------------------

The only dependencies for this code, are 

(1) A Python interpreter (http://www.python.org/).  The code has been tested with Python 2.7.5

and

(2) NumPy -- the fundamental package for scientific computing with Python (http://www.numpy.org/).  The code has been tested with NumPy 1.7.1

An easy approach to get up and running with Python and NumPy is to install the Anaconda distribution from http://continuum.io/downloads
