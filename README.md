# Iterative Matrix Pencil Method - Python Version
Last Updated: July 26, 2022

This is a Python implementation of the Iterative Matrix Pencil Method, found [here.](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/17ec96d0-925a-4752-ac55-e7a83d97fe65/content)

The code is broken up into 4 components: 

```
Iterative Matrix Pencil Method
└─── options.py
└─── detrend.py
└─── IMP.py
└─── main.py
```

## options.py
This file contains the input options for the Iterative Matrix Pencil method, and are currently set to their default values. This is read in by `IMP.py` and considers the following parameters:

* SVD Threshold
* Number of Iterations 
* Initial Starting Signal
* Detrending Type
* IMP Type (Random or Non-Random)

## detrend.py
This file has different versions of detrending (constant, linear, quadratic), and a generalized polynomial fit to include higher degree detrendings.

## IMP.py
This contains the bulk of the components for the Iterative Matrix Pencil method code. Includes the following components of the Iterative Matrix Pencil Method:

* Matrix Pencil Method
* Calculation of Mode Shapes
* Re-construction of the Signals
* Calculation of Cost Function
* Presentation of Modes

## main.py
This contains the file input requests and the following components and variations of the Iterative Matrix Pencil method.

* Standard Iterative Matrix Pencil Method

A single run of the IMP method, with options set in the `options.py` file. 

* Random Iterative Matrix Pencil Method (commented out)

300 runs of the Iterative Matrix Pencil method, but with random signal selection for each iteration. 

* Varying Iterative Matrix Pencil Method

Multiple runs of the IMP method, where you can vary any one user-input over a range of values. 
