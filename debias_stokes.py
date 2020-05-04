"""
Background:

Debiasing script based on Chat Hull's scripts, updated to Python 3. Does not
take into account any sort of primary beam correction which must be applied
afterwards.

Useage:

Should be a simple command line procedure:

$ python debias_stokes.py -Q path/to/stokes_Q.fits -U path/to/stokes_U.fits

This will estimate the noise based on the Q and U maps, then save a debiased
map to `path/to/debiased.fits`. In the odd case that Q and U components are in
different folders, it will use the folder of the Q component.

Additional arguments can be found using:

$ python debias_stokes.py --help

Requires:

    - argparse
    - astropy
    - gofish
    - numpy
    - scipy

All of which should be pip installable.
"""
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import iv
from gofish import imagecube
from astropy.io import fits
import numpy as np
import argparse


def _PDF(SNR_true, SNR_obs, z_max=1000.0):
    """
    The probability distribution function for p(Ptrue | Pobs, rms). All
    arguments should share the same units.

    Args:
        SNR_true (array): The true, polarization signal-to-noise ratio.
        SNR_obs (float): The observed polarization signal-to-noise ratio.

    Returns:
        PDF (float): The probability distribution function.
    """
    z = SNR_obs * SNR_true
    P = SNR_obs * iv(0, np.where(z > z_max, 0.0, z))
    return P * np.exp(-(SNR_obs**2 + SNR_true**2) / 2)


def _nPDF(SNR_true, SNR_obs, z_max=1000.0):
    """The negative of the PDF for minimization."""
    return -1.0 * _PDF(SNR_true, SNR_obs, z_max)


def _interpolate_p_SNR(SNR_obs=None, minimize_kwargs=None,
                       interp1d_kwargs=None):
    """
    Returns a interp1d instance to interpoalte between a pre-calculated grid of
    {p_obs, sigma} -> p_true transformations. The defaults for this function
    allow for a good coverage of the expected `p_obs`.

    Args:
        SNR_obs (optional[array]): Array of observed polarization SNRs.
        minimize_kwargs (optional[dict]): Dictionary of keyword arguments to
            pass to `minimize`.
        interp1d_kwrgs (optional[dict]): Dictionary of keyword arguments to
            pass to `interp1d`.

    Returns:
        SNR (callable): An `interp1d` instance which will return the true
            polarized SNR ratio given an observed polarized SNR.
    """
    # Default range of SNRs expected from data.
    SNR_obs = np.logspace(0, 1, 200) if SNR_obs is None else SNR_obs

    # Find the values which maximize the PDF for each SNR_obs.
    minimize_kwargs = {} if minimize_kwargs is None else minimize_kwargs
    minimize_kwargs['method'] = minimize_kwargs.pop('method', 'L-BFGS-B')
    minimize_kwargs['tol'] = minimize_kwargs.pop('tol', 1e-12)
    minimize_kwargs['bounds'] = minimize_kwargs.pop('bounds', [(0.0, None)])
    SNR_true = [minimize(_nPDF, x0=[obs], args=(obs), **minimize_kwargs).x
                for obs in SNR_obs]
    SNR_true = np.squeeze(SNR_true)

    # Create the interp1d instance and return.
    interp1d_kwargs = {} if interp1d_kwargs is None else interp1d_kwargs
    interp1d_kwargs['fill_value'] = 'extrapolate'
    interp1d_kwargs['bounds_error'] = False
    return interp1d(SNR_obs, SNR_true, **interp1d_kwargs)


def _output_path(path, new_path='debaised_PI.fits'):
    """Returns the new output path."""
    return path.replace(path.split('/')[-1], new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Q', type=str,
                        help='Path to the Stokes Q FITS cube.')
    parser.add_argument('-U', type=str,
                        help='Path to the Stokes U FITS cube.')
    parser.add_argument('-rms', type=float,
                        help='RMS of the Stoke Q and U cubes.')
    parser.add_argument('-overwrite', default=True, type=bool,
                        help='Overwrite existing files with the same name.')
    args = parser.parse_args()

    # Read in the cubes.
    if args.Q is not None:
        cube_Q = imagecube(args.Q)
    else:
        raise ValueError("Must specify a Stokes Q cube.")
    if args.U is not None:
        cube_U = imagecube(args.U)
    else:
        raise ValueError("Must specify a Stokes U cube.")

    # Calculate the P_obs values.
    P_obs = np.hypot(cube_Q.data, cube_U.data)
    if args.rms is None:
        print("Estimating QU rms based on first and last 15 channels.")
        rms = np.mean([cube_Q.estimate_RMS(15), cube_U.estimate_RMS(15)])
        print("sigma = {:.2f} mJy/beam".format(rms * 1e3))
    else:
        rms = args.rms
    SNR_obs = P_obs / rms
    if SNR_obs.shape != cube_Q.data.shape:
        raise ValueError("Wrong shape for `SNR_obs`: {}".frmat(SNR_obs.shape))

    # Calculate the interpolator and debias the cubes.
    true_SNR_interp = _interpolate_p_SNR()
    P_true = np.array([true_SNR_interp(c) * rms for c in SNR_obs])

    # Save as a FITS file. Copying the header from one of the input cubes.
    header = fits.getheader(cube_Q.path, copy=True)
    header['comment'] = 'debiased polarized intensity'
    fits.writeto(_output_path(args.Q), P_true, header,
                 overwrite=args.overwrite, output_verify='silentfix')
