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
from scipy.special import i0
from gofish import imagecube
from scipy.stats import rice
from astropy.io import fits
import numpy as np
import argparse


def _PDF(A, B):
    """
    The Rice probability distribution function for p(A|B).
    """
    return rice.pdf(A, B)


def _nPDF(A, B):
    """
    The negative of the Rice PDF.
    """
    return -_PDF(A, B)


def _CDF(A, B):
    """
    The Rice cumulative distribution function for p(A|B).
    """
    return rice.cdf(A, B)


def _PPF(q, B):
    """
    For the CDF of the Rice distribution, p(A|B), returns the value of A which
    yields the quantile q. This is known as the 'percent point function'.
    """
    return rice.ppf(q, B)


def _L(P0, P):
    """
    The likelihood of P0 given P, L(P0|P), from Vaillancourt 2006.
    """
    L = _PDF(P, P0) / P / np.sqrt(np.pi / 2.0)
    return L / np.exp(-P**2 / 4.0) / i0(P**2 / 4.0)


def _nL(P0, P):
    """
    The negative of the likelihood of P0 given P.
    """
    return -_L(P0, P)


def _maxP(P0, P):
    """
    Function to minimize to infer P0 such that ``max(F(P|P0)) == P``.
    """
    maxP = minimize(_nPDF, x0=P0, args=P0, method='Nelder-Mead')
    return abs(maxP.x[0] - P)


def _infer_dP0(dP0, P, P0, quantile=0.68):
    """
    Find the dP0 which achieves the desired quantile.
    """
    model = _PDF(P0 + dP0, P) - _PDF(P0 - dP0, P)
    return abs(model - quantile)


def infer_P0(P, estimator='probable', quantile=0.68):
    """
    For a given `P` (defined as `hypot(Q, U) / sigma`), infer the true
    polarization, `P0` and the associated uncertainty, `dP0`, defined as the
    range `P0 +/- dP0` which encompasses 68% of the distribution F(P0|P).

    Two estimators are possible, for the inference of `P0`, the maximum
    likelihood, ``estimator='likelihood'``, or the most probable,
    ``estimator='probable'``.

    For each of these, if `P < threshold`, where the thresholds are `1` for the
    most probable estimator and `sqrt(2)` for the maximum likelihood estimator,
    `P0 = 0`.

    Args:
        P (float): The observed polarization value normalised by the
            uncertainty measured as the Q/U RMS.
        estimator (optional[str]): The estimator used for the inference of
            `P0`, either ``'probable'``, the default, or ``'likelihood'``.
        quantile (optional[float]): The quantile used to define the
            uncertainties on ``P0``. The default, 0.68, is equivlant to 1
            sigma for Normal distributions.

    Returns:
        P0, dP0 (float, float): The inferred true polarization and uncertainty.
    """

    # Check the estimator.
    if estimator == 'likelihood':
        func = _nL if P > np.sqrt(2.0) else None
    elif estimator == 'probable':
        func = _maxP if P > 1.0 else None
    else:
        raise ValueError("Unknown estimator: '{}'.".format(estimator))

    # Calculate the underlying polarization, P0.
    if func is not None:
        P0 = minimize(func, x0=P, args=P, method='Nelder-Mead')
        if not P0.success:
            return np.nan, np.nan
        P0 = P0.x[0]
    else:
        P0 = 0.0
    P0 = 0.0 if np.isclose(P0, 0.0) else P0

    # Estimate the uncertainties.
    # TODO: Explore better ways to deal with upper limits only.
    dP0 = minimize(_infer_dP0, x0=1.0, args=(P, P0), method='Nelder-Mead').x[0]

    # Return
    return P0, dP0


def _get_interp_P0(P=None, estimator='probable', interp1d_kwargs=None):
    """
    Returns two ``interp1d`` instances to interpoalte between a pre-calculated
    grid of `P -> P0` and `P -> dP0` transformations. The defaults for this
    function allow for a good coverage of the expected `P` values.

    TODO: Allow kwargs to be passed to scipy.optimize.minimize.

    Args:
        P (optional[array]): Array of observed polarization values normalized
            by the RMS measured in the Q and U components.
        estimator (optional[str]): Estimator to use for the inference of `P0`.
        interp1d_kwrgs (optional[dict]): Dictionary of keyword arguments to
            pass to `interp1d`.

    Returns:
        P0, dP0 (callable, callable): ``interp1d`` instances returning the
            interpolated ``P0`` and ``dP0`` values for a given ``P``.
    """
    P = np.linspace(1e-5, 10, 200) if P is None else P
    P0, dP0 = np.array([infer_P0(PP, estimator=estimator) for PP in P]).T
    ik = {} if interp1d_kwargs is None else interp1d_kwargs
    ik['fill_value'], ik['bounds_error'] = 'extrapolate', False
    return interp1d(P, P0, **ik), interp1d(P, dP0, **ik)


def _output_path(path, new_path='P0.fits'):
    """Returns the new output path."""
    return path.replace(path.split('/')[-1], new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Q', type=str,
                        help='Path to the Stokes Q FITS cube.')
    parser.add_argument('-U', type=str,
                        help='Path to the Stokes U FITS cube.')
    parser.add_argument('-N', default=15, type=int,
                        help='Number of first and last channels to use when '
                             + 'estimating the RMS of the Stokes Q and U '
                             + 'cubes.')
    parser.add_argument('-estimator', default='probable', type=str,
                        help='Estimator to use for the debiasing. Either '
                             + '`probable` or `likelihood`.')
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
    if args.rms is None and P_obs.ndim == 3:
        rms = np.mean([cube_Q.estimate_RMS(args.N),
                       cube_U.estimate_RMS(args.N)])
        print("Estimated Q/U RMS based on the first and last "
              + "{} channels: ".format(args.N)
              + "{:.2f} mJy/beam.".format(rms * 1e3))
    elif args.rms is None and P_obs.ndim == 2:
        rms = np.nanstd([cube_Q.data, cube_U.data])
        print("Estimated Q/U RMS based on the full image: "
              + "{:.2f} mJy/beam.\n".format(rms * 1e3)
              + "Try with the '-rms' flag for more accurate results.")
    else:
        rms = args.rms
    SNR_obs = P_obs / rms

    # Check shapes.
    if SNR_obs.shape != cube_Q.data.shape:
        raise ValueError("Wrong shape for `SNR_obs`: {}".frmat(SNR_obs.shape))
    if SNR_obs.ndim != 3:
        SNR_obs = np.array([SNR_obs])

    # Calculate the interpolator and debias the cubes.
    print("Building interpolation tables...")
    interp_P0, interp_dP0 = _get_interp_P0(estimator=args.estimator)
    print("Interpolating pixel values...")
    P0 = np.array([interp_P0(c) * rms for c in SNR_obs])
    dP0 = np.array([interp_dP0(c) * rms for c in SNR_obs])

    # Save as a FITS file. Copying the header from one of the input cubes.
    header = fits.getheader(cube_Q.path, copy=True)
    header['comment'] = 'debiased polarized intensity'
    fits.writeto(_output_path(args.Q, 'P0.fits'), P0, header,
                 overwrite=args.overwrite, output_verify='silentfix')

    header = fits.getheader(cube_Q.path, copy=True)
    header['comment'] = 'debiased polarized intensity uncertainty'
    fits.writeto(_output_path(args.Q, 'dP0.fits'), dP0, header,
                 overwrite=args.overwrite, output_verify='silentfix')
