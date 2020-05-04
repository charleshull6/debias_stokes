# debias_stokes.py

Quick script to debias polarization observations using the approach outlined in [Hull & Plambeck (2015)](https://ui.adsabs.harvard.edu/abs/2015JAI.....450005H/abstract). The docstring should have sufficient information to run, but in brief:

```bash
$ python debias_stokes.py -Q path/to/stokes_Q.fits -U path/to/stokes_U.fits
```

will spit out a new file at `path/to/debiased.fits` which will contain the debaised polarization in the same units as the provided image cubes.

For more help use

```bash
$ python debiase_stokes.py --help
```
