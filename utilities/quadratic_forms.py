###############################################################################
# initial imports:

import math

import numpy as np
from numba import njit
import scipy.integrate as integrate
import scipy.fft as fft
import scipy.interpolate as interpolate
import scipy.stats as stats

import tensiometer.utilities as utilities


###############################################################################
# cumulants:

def cumulants(eigenvalues, n, multiplicity=None):
    """
    Get cumulants for quadratic form
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    # get multiplicity:
    if multiplicity is None:
        _multiplicity = np.ones(len(eigenvalues))
    else:
        _multiplicity = np.array(multiplicity)
    # compute:
    _res = 2**(n-1) * math.factorial(n-1) * np.sum(_multiplicity * eigenvalues**n)
    #
    return _res

###############################################################################
# Patnaiks approximation 1:


def patnaik_1(eigenvalues, multiplicity=None):
    """
    Returns the CDF for the Patnaiks approximation
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    # get dofs:
    if multiplicity is not None:
        multiplicity = np.array(multiplicity)
        dofs = np.sum(multiplicity*eigenvalues)
    else:
        dofs = np.sum(eigenvalues)
    #
    return stats.chi2(dofs)

###############################################################################
# Patnaiks approximation 2:


def patnaik_2(eigenvalues, multiplicity=None):
    """
    Returns the CDF for the Patnaiks approximation
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    # get dofs:
    if multiplicity is not None:
        multiplicity = np.array(multiplicity)
        c = np.sum(multiplicity*eigenvalues**2) / np.sum(multiplicity*eigenvalues)
        nu = np.sum(multiplicity*eigenvalues)**2 / np.sum(multiplicity*eigenvalues**2)
    else:
        c = np.sum(eigenvalues**2) / np.sum(eigenvalues)
        nu = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    #
    return stats.gamma(a=0.5*nu, scale=2.*c)

###############################################################################
# Monte Carlo CDF method:


def cdf_monte_carlo(eigenvalues, Qobs, multiplicity=None, atol=1.e-3, batch_size=100, max_batches=1000, **kwargs):
    """
    Brute force monte carlo method. Central quadratic forms, not necessarily positive definite.
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    # get multiplicity:
    if multiplicity is not None:
        _eigenvalues = np.repeat(eigenvalues, multiplicity)
    else:
        _eigenvalues = eigenvalues
    # get number of components:
    num_components = len(_eigenvalues)
    # do the batched MC:
    num_pass = 0
    num_test = 0
    for i in range(max_batches):
        # generate:
        _temp = np.random.normal(size=num_components*batch_size).reshape((num_components, batch_size))
        # test and count:
        _temp = np.dot(_eigenvalues, _temp**2)
        num_pass += np.sum(_temp > Qobs)
        num_test += batch_size
        # compute error:
        _min, _max = utilities.clopper_pearson_binomial_trial(num_pass, num_test, **kwargs)
        if (_max - _min) / 2. < atol:
            break
    #
    return num_pass / num_test, utilities.clopper_pearson_binomial_trial(num_pass, num_test, **kwargs)


###############################################################################
# Imhof method for quadratic form CDF:

# define first part of integrand:
@njit("float64(float64,float64[::1],float64,float64[::1])")
def theta(u, eigenvalues, Qobs, multiplicity):
    return 0.5*(np.sum(multiplicity*np.arctan(eigenvalues*u))-Qobs*u)

# define second part of integrand:
@njit("float64(float64,float64[::1],float64[::1])")
def rho(u, eigenvalues, multiplicity):
    return np.prod((1.0+eigenvalues**2*u**2)**(0.25*multiplicity))

# define integrand:
@njit("float64(float64,float64[::1],float64,float64[::1])")
def integrand(u, eigenvalues, Qobs, multiplicity):
    if u == 0.0:
        return 0.5*(np.sum(multiplicity*eigenvalues) - Qobs)
    else:
        return np.sin(theta(u, eigenvalues, Qobs, multiplicity))/u/rho(u, eigenvalues, multiplicity)

# define integrand for ode solver method:
@njit("float64(float64[::1],float64,float64[::1],float64,float64[::1])")
def integrand_ode_1(t, u, eigenvalues, Qobs, multiplicity):
    if u == 0.0:
        return 0.5*(np.sum(multiplicity*eigenvalues) - Qobs)
    else:
        return np.sin(theta(u, eigenvalues, Qobs, multiplicity))/u/rho(u, eigenvalues, multiplicity)

# define integrand for ode solver method after the first step:
@njit("float64(float64[::1],float64,float64[::1],float64,float64[::1])")
def integrand_ode_2(t, u, eigenvalues, Qobs, multiplicity):
    return np.sin(theta(u, eigenvalues, Qobs, multiplicity))/u/rho(u, eigenvalues, multiplicity)


def cdf_imhof(eigenvalues, Qobs, multiplicity=None, atol=1.e-3, method='ode', **kwargs):
    """
    Imhof method. Central quadratic forms, not necessarily positive definite.
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    _n = len(eigenvalues)
    # get multiplicity:
    if multiplicity is None:
        _multiplicity = np.ones(_n)
    else:
        _multiplicity = np.array(multiplicity)
    # rescale the problem so that the integration region is between 0 and 1:
    _k = 0.5*np.sum(_multiplicity)
    _factor = -np.log(atol*np.pi*_k)/_k - 0.5 / _k * np.sum(_multiplicity*np.log(np.abs(eigenvalues)))
    _eigenvalues = eigenvalues * np.exp(_factor)
    _Qobs = Qobs * np.exp(_factor)
    _max_U = 1.0
    # compute number of oscillation periods:
    _num_periods = _max_U * np.abs(_Qobs) / 4. / np.pi
    # adjust tollerance for integrator (we ask to be better than truncation by a factor 10)
    epsabs = atol / 10.
    # do the integral:
    if method == 'quad':
        # compute maximum number of subdivisions:
        _sub = max(50, 10*2*int(np.ceil(_num_periods)))
        # do the integral:
        _res, _err = integrate.quad(integrand, 0, _max_U, args=(_eigenvalues, _Qobs, _multiplicity), epsabs=epsabs, epsrel=epsabs, limit=_sub, **kwargs)
    elif method == 'ode':
        # compute boundary conditions:
        _y0 = 0.5*(np.sum(_multiplicity*_eigenvalues) - _Qobs)
        # compute initial grid (which is adaptively refined internally) based on oscillation rate of the frequency term
        _num_grid = max(2*int(np.ceil(_num_periods)), 10)
        _grid = np.linspace(0, _max_U, _num_grid)
        # do the first step:
        _res_1 = integrate.odeint(integrand_ode_1, _y0, _grid[0:2], args=(_eigenvalues, _Qobs, _multiplicity), rtol=epsabs, atol=epsabs, **kwargs)
        # do the other steps:
        _res_2 = integrate.odeint(integrand_ode_2, _res_1[-1][0], _grid[1:], args=(_eigenvalues, _Qobs, _multiplicity), rtol=epsabs, atol=epsabs, **kwargs)
        # combine:
        _res = _res_2[-1][0] - _y0
        _err = max(1.e-8, epsabs)  # lsoda is not double precision...
    # process output:
    _res = 0.5+_res/np.pi
    _err = _err/np.pi + atol
    #
    return _res, _err

###############################################################################
# Imhof method for quadratic form CDF with oscillatory integrator:


# define b(u):
@njit("float64(float64,float64[::1],float64[::1])")
def bu(u, eigenvalues, multiplicity):
    return 0.5*np.sum(multiplicity*np.arctan(eigenvalues*u))

# define first integrand:
@njit("float64(float64,float64[::1],float64[::1])")
def osc_integrand_1(u, eigenvalues, multiplicity):
    return np.sin(bu(u, eigenvalues, multiplicity))/u/rho(u, eigenvalues, multiplicity)

# define second integrand:
@njit("float64(float64,float64[::1],float64[::1])")
def osc_integrand_2(u, eigenvalues, multiplicity):
    return np.cos(bu(u, eigenvalues, multiplicity))/u/rho(u, eigenvalues, multiplicity)


def cdf_imhof_osc(eigenvalues, Qobs, multiplicity=None, atol=1.e-3, **kwargs):
    """
    Imhof method. Central quadratic forms, not necessarily positive definite.
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    _n = len(eigenvalues)
    # get multiplicity:
    if multiplicity is None:
        _multiplicity = np.ones(_n)
    else:
        _multiplicity = np.array(multiplicity)
    # rescale so that the problem has mean one (in the positive case):
    _factor = np.sum(_multiplicity * np.abs(eigenvalues))
    _eigenvalues = eigenvalues / _factor
    _Qobs = Qobs / _factor
    # adjust error control:
    epsrel = atol / 3.
    # discriminate the cases when Qobs is zero or not:
    if _Qobs == 0.:
        # do a first (tiny) step with standard integrator so that we avoid dealing with the singularity at the origin:
        _upper = np.pi
        _res_1, _err_1 = integrate.quad(integrand, 0, _upper, args=(_eigenvalues, _Qobs, _multiplicity), epsabs=epsrel, epsrel=epsrel, **kwargs)
        # then do the infinite integrals:
        _res_2, _err_2 = integrate.quad(osc_integrand_1, _upper, np.inf, args=(_eigenvalues, _multiplicity),
                                        epsabs=epsrel, epsrel=epsrel, **kwargs)
        _res_3, _err_3 = integrate.quad(osc_integrand_2, _upper, np.inf, args=(_eigenvalues, _multiplicity),
                                        epsabs=epsrel, epsrel=epsrel, **kwargs)
    else:
        # do a first (tiny) step with standard integrator so that we avoid dealing with the singularity at the origin:
        _upper = np.pi / np.abs(_Qobs)
        _res_1, _err_1 = integrate.quad(integrand, 0, _upper, args=(_eigenvalues, _Qobs, _multiplicity), epsabs=epsrel, epsrel=epsrel, **kwargs)
        # do the first oscillatory integral:
        _res_2, _err_2 = integrate.quad(osc_integrand_1, _upper, np.inf, args=(_eigenvalues, _multiplicity),
                                        weight='cos', wvar=0.5*_Qobs, epsabs=epsrel, epsrel=epsrel, **kwargs)
        _res_3, _err_3 = integrate.quad(osc_integrand_2, _upper, np.inf, args=(_eigenvalues, _multiplicity),
                                        weight='sin', wvar=0.5*_Qobs, epsabs=epsrel, epsrel=epsrel, **kwargs)
    _res = _res_1 + _res_2 - _res_3
    # process output:
    _res = 0.5+_res/np.pi
    _err = (_err_1 + _err_2 + _err_3)/np.pi
    #
    return _res, _err


###############################################################################
# Imhof method for quadratic form PDF with oscillatory integrator:

# define second integrand:
@njit("float64(float64,float64[::1],float64[::1])")
def pdf_osc_integrand_1(u, eigenvalues, multiplicity):
    return np.cos(bu(u, eigenvalues, multiplicity))/rho(u, eigenvalues, multiplicity)

# define first integrand:
@njit("float64(float64,float64[::1],float64[::1])")
def pdf_osc_integrand_2(u, eigenvalues, multiplicity):
    return np.sin(bu(u, eigenvalues, multiplicity))/rho(u, eigenvalues, multiplicity)


def pdf_imhof_osc(eigenvalues, Q, multiplicity=None, atol=1.e-3, **kwargs):
    """
    Imhof method. Central quadratic forms, not necessarily positive definite.
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    _n = len(eigenvalues)
    # get multiplicity:
    if multiplicity is None:
        _multiplicity = np.ones(_n)
    else:
        _multiplicity = np.array(multiplicity)
    # rescale so that the problem has mean one (in the positive case):
    _factor = np.sum(_multiplicity * np.abs(eigenvalues))
    _eigenvalues = eigenvalues / _factor
    Q = Q / _factor
    # adjust error control:
    epsrel = atol / 3.
    # we need to distinguish the Q=0 case from others:
    if Q == 0.:
        # do the first oscillatory integral:
        _res_1, _err_1 = integrate.quad(pdf_osc_integrand_1, 0.0, np.inf, args=(_eigenvalues, _multiplicity),
                                        epsabs=epsrel, epsrel=epsrel, **kwargs)
        # do the second oscillatory integral:
        _res_2, _err_2 = 0., 0.
    else:
        # do the first oscillatory integral:
        _res_1, _err_1 = integrate.quad(pdf_osc_integrand_1, 0.0, np.inf, args=(_eigenvalues, _multiplicity),
                                        weight='cos', wvar=0.5*Q, epsabs=epsrel, epsrel=epsrel, **kwargs)
        # do the second oscillatory integral:
        _res_2, _err_2 = integrate.quad(pdf_osc_integrand_2, 0.0, np.inf, args=(_eigenvalues, _multiplicity),
                                        weight='sin', wvar=0.5*Q, epsabs=epsrel, epsrel=epsrel, **kwargs)
    #
    _res = _res_1 + _res_2
    # process output:
    _res = 0.5 / np.abs(_factor) * _res / np.pi
    _err = 0.5 / np.abs(_factor) * (_err_1 + _err_2)/np.pi
    #
    return _res, _err


###############################################################################
# Quadratic form PDF with FFT:


# define the fft integrand:
@njit("float64[::1](float64[::1],float64[::1],float64[::1])")
def pdf_fft_integrand(u, eigenvalues, multiplicity):
    _res = np.zeros(len(u))
    for i in range(len(u)):
        _res[i] = np.cos(bu(2.*u[i], eigenvalues, multiplicity))/rho(2.*u[i], eigenvalues, multiplicity) +np.sin(bu(2.*u[i], eigenvalues, multiplicity))/rho(2.*u[i], eigenvalues, multiplicity)
    return _res


def pdf_fft(eigenvalues, Q, multiplicity=None, num_points=2**12, full_results=False, expand=0, **kwargs):
    """
    Quadratic form pdf with FFT algorithm.
    """
    # process input:
    eigenvalues = np.array(eigenvalues)
    _n = len(eigenvalues)
    # get multiplicity:
    if multiplicity is None:
        _multiplicity = np.ones(_n)
    else:
        _multiplicity = np.array(multiplicity)
    # rescale so that the problem has mean one (in the positive case):
    _factor = np.sum(_multiplicity * np.abs(eigenvalues))
    _eigenvalues = eigenvalues / _factor
    # make sure number of FFT points is rounded to the closest multiple of two:
    _num_points = 2**int(np.ceil(np.log2(num_points)))
    # get the maximum Q, we always need to resolve the entire distribution to do FFT
    _Q_ext = max(np.abs(np.amin(Q)), np.abs(np.amax(Q))) / _factor
    _min_Q_ext = cumulants(_eigenvalues, 1) + 6*np.sqrt(cumulants(_eigenvalues, 2))
    _Q_ext = max(_Q_ext, _min_Q_ext)
    _Q_ext = 10**expand * _Q_ext
    # prepare the u and Q grids:
    _Q_spacing = 2.*_Q_ext / _num_points
    _u_spacing = np.pi / _Q_ext
    _u_grid = fft.fftshift(fft.fftfreq(_num_points, _Q_spacing / (2.*np.pi)))
    _Q_grid = fft.fftshift(fft.fftfreq(_num_points, _u_spacing / (2.*np.pi)))
    # compute FFT integrand:
    _integrand = pdf_fft_integrand(_u_grid, _eigenvalues, _multiplicity)
    # do the FFT:
    _fft = fft.fftshift(fft.fft(_integrand))
    # combine:
    _res = np.abs(np.real(_fft) - np.imag(_fft))
    # normalization factor:
    _res = _res / np.trapz(_res, _Q_grid) / np.abs(_factor)
    _Q_grid = np.abs(_factor) * _Q_grid
    # now interpolate and evaluate on the input grid:
    _interpolator = interpolate.interp1d(_Q_grid, _res)
    _interp_res = _interpolator(Q)
    # return:
    if full_results:
        return _interp_res, _interpolator
    else:
        return _interp_res
