# ld-exosim

This repository stores codes to calculate the limb-darkening induced biases on various exoplanet 
parameters, as explained in Espinoza & Jordán (2015b). 

DEPENDENCIES
------------

This code makes use of three important libraries:

    + The Bad-Ass Transit Model cAlculatioN (batman) package: http://astro.uchicago.edu/~kreidberg/batman/
    + The latest version of the lmfit fitter (https://lmfit.github.io/lmfit-py/)
    + The LDC3.py code wrote by David Kipping (https://github.com/davidkipping/LDC3)

This last code might be updated with time, but I have copied here the October 29th, 2015 version of it
for reference: be sure to use the latest version of D. Kipping's code!

USAGE
------------
The usage of the code is simple: just define the parameters you woud like to explore in the run_ld_exosim.py 
code and run it. The results will be saved in a folder named "results" for your simulation. 

The code makes use of limb-darkening tables stored in the ld_tables folder. To generate your own, you can 
use our code in https://github.com/nespinoza/limb-darkening and put the result inside.

