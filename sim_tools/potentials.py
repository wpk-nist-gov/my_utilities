import numpy as np
from scipy.optimize import minimize, fsolve
import click

def phi_lennardjones(r,
                     sig=1.0,
                     eps=1.0,
                     rcut=np.inf,
                     forceshift=False,
                     ret_dphi=False):
    """
    Lennard jones potential

    Parameters
    ----------
    r : float or array-like

    sig, eps : float
    LJ parameters

    rcut : float
    cut off for potential

    forceshift : bool (Default False)
    if True, force shift potential

    ret_dphi : bool (Default False)
    if True, return dphi (see below)


    Returns
    -------
    phi : same type as r
    LJ potentail at positions r

    dphi : same type as r
    -r**(-1)*d(phi)/d(r).  
    """
    r = np.asarray(r)
    V = np.full_like(r, fill_value=np.inf)

    m = r > 0
    x = sig / r[m]

    V[m] = 4.0 * eps * (x**12 - x**6)

    if ret_dphi:
        dphi = np.full_like(r, fill_value=np.inf)
        dphi[m] = 48 * eps * (x**12 - 0.5 * x**6) / r[m]**2  #= -dvdrcut/r

    if np.isfinite(rcut):
        Vcut, dphicut = phi_lennardjones(
            rcut, sig, eps, np.inf,
            False, ret_dphi=True)
        msk = r <= rcut

        V[msk] -= Vcut
        V[~msk] = 0.
        if ret_dphi:
            dphi[~msk] = 0.

        if forceshift is True:
            # xx = sig / rcut
            # dVdrcut = -48. * eps * (xx**12 - 0.5 * xx**6) / rcut
            # dphicut = -dVdrcut/rcut

            dVdrcut = -dphicut * rcut
            V[msk] -= dVdrcut * (r[msk] - rcut)
            if ret_dphi:
                dphi[msk] += dVdrcut/r[msk] #=-dphicut*rcut/r[msk]

    # if r,V are 0-D arrays, convert to floats
    V *= 1

    if ret_dphi:
        dphi *= 1
        return V, dphi
    else:
        return V


def phi_LJ(r, sig=1.0, eps=1.0):
    """
    Lennard jones potential

    Parameters
    ----------
    r : float or array-like

    sig, eps : float
    LJ parameters

    Returns
    -------
    phi : same type as r
    LJ potentail at positions r

    dphi : same type as r
    -r**(-1)*d(phi)/d(r).  
    """
    r = np.asarray(r)
    x = sig / r
    V = 4.0 * eps * (x**12 - x**6)
    dphi = 48 * eps * (x**12 - 0.5 * x**6) / r**2  #-dvdrcut/r

    return V, dphi


import functools

def phi_cut(f, rcut, forceshift=False):
    """
    return cut and shifted form of potential f(r,**kwargs)

    Parameters
    ----------
    f : function
        f(r,**kwargs) should return (phi, dphi) at r, where
        dphi = - 1/r * df/dr

    forceshift : bool
        if True, apply force shift 

    kwargs : extra arguments to f

    Returns
    -------
    f_cut : cut potential
    """

    @functools.wraps(f)
    def _phi(r, *args, **kwargs):
        V, dphi = f(r, *args, **kwargs)
        V = np.asarray(V)
        dphi = np.asarray(dphi)

        # cut/shift
        Vcut, dphicut = f(rcut, *args, **kwargs)
        msk = r <= rcut

        V[msk] -= Vcut
        V[~msk] = 0.
        dphi[~msk] = 0.

        if forceshift:
            dVdrcut = -dphicut * rcut

            V[msk] -= dVdrcut * (r[msk] - rcut)
            dphi[msk] += dVdrcut/r[msk] #-dphicut*rcut/r[msk]

        V = V * 1.
        dphi = dphi * 1.
        return V, dphi

    return _phi



from scipy.integrate import quad
def B2_pot(phi, b, a=0., beta=1., quad_kwargs=None, ret_error=False, **kwargs):
    """
    second virial for potential function phi

    Parameters
    ----------
    phi : function
        phi(r,**kwargs) returns potential at r

    a, b : floats
        integration bounds

    beta : float
        inverse thermal energy (kT)**(-1)

    quad_kwargs : dict
        extra arguments to scipy.integrate.quad

    **kwargs : keyword arguments
        extra kwargs to phi

    Returns
    -------
    B2 : float
        second virial

    err : float
        integration error in B2 calculation
    """

    f = lambda x: x**2 * (np.exp(-beta * phi(x, **kwargs)) - 1.)

    quad_kwargs = quad_kwargs or {}
    integral, error = quad(f, a, b, **quad_kwargs)
    fac = 2. * np.pi
    if ret_error:
        return -fac * integral, fac * error
    else:
        return -fac * integral


def dB2dbeta(phi, b, a=0.0, beta=1.0, quad_kwargs=None, ret_error=False, **kwargs):
    """
    derivate second virial for potential function phi w.r.t beta

    Note that T * dB/dT = -beta dB/dbeta

    Parameters
    ----------
    phi : function
        phi(r,**kwargs) returns potential at r

    a, b : floats
        integration bounds

    beta : float
        inverse thermal energy (kT)**(-1)

    quad_kwargs : dict
        extra arguments to scipy.integrate.quad

    **kwargs : keyword arguments
        extra kwargs to phi

    Returns
    -------
    B2 : float
        second virial

    err : float
        integration error in B2 calculation
    """

    def _f(x):
        v = phi(x, **kwargs)
        return x**2 * v * np.exp(-beta * v)
    quad_kwargs = quad_kwargs or {}
    integral, error = quad(_f, a, b, **quad_kwargs)
    fac = 2. * np.pi
    if ret_error:
        return fac * integral, fac * error
    else:
        return fac * integral

class PotentialAnalysis(object):
    """
    split a pair potential into 
    """


    def __init__(self, pot, x0=None, bounds=None, UB=None, **kwargs):
        self._pot = pot
        self._x0 = x0
        self._UB = UB
        self._bounds = bounds
        self._kwargs = kwargs


    def pot(self, x):
        return self._pot(x)

    def set_min(self, x0=None, bounds=None, **kwargs):
        x0 = x0 or self._x0
        assert x0 != None
        bounds = bounds or self._bounds
        assert bounds != None
        bounds = [bounds]

        kwars = dict(self._kwargs, **kwargs)
        self._params = minimize(self.pot, x0=x0, bounds=bounds, **kwargs)

    @property
    def params(self):
        if not hasattr(self,'_params'):
            self.set_min()
        return self._params

    @property
    def x_min(self):
        return self.params.x[0]

    @property
    def pot_min(self):
        return self.params.fun[0]

    def pot_rep(self, x):
        x = np.asarray(x)
        V = self.pot(x)
        V = np.asarray(V)
        msk = x <= self.x_min
        V[msk] -= self.pot_min
        V[~msk] = 0
        return V*1.0

    def pot_att(self, x):
        x = np.asarray(x)
        V = self.pot(x)
        V = np.asarray(V)
        msk = x >= self.x_min
        V[msk] -= self.pot_min
        V[~msk] = 0

        return V*1.0


    def sigma_BH(self, beta, ret_error=False, **kwargs):
        f = lambda x: 1.0 - np.exp(-beta * self.pot_rep(x))
        integral, error = quad(f, 0.0, self.x_min, **kwargs)

        if ret_error:
            return integral, error
        else:
            return integral


    def B2(self, beta, UB=None, LB=0.0, ret_error=False, quad_kwargs=None, **kwargs):
        UB = UB or self._UB
        assert UB != None
        return B2_pot(self.pot, b=UB, a=LB, beta=beta,
                      quad_kwargs=quad_kwargs,
                      ret_error=ret_error,
                      **kwargs)

    def dB2dbeta(self, beta, UB=None, LB=0.0, ret_error=False, quad_kwargs=None, **kwargs):
        UB = UB or self._UB

        assert UB != None

        return dB2dbeta(self.pot, b=UB, a=LB, beta=beta, quad_kwargs=quad_kwargs, ret_error=ret_error, **kwargs)


    def corresponding_states(self, beta, UB=None, LB=0.0, quad_kwargs=None, **kwargs):
        """
        find corresponding state sigma, lambda, epsilon for square well fluid.

        sigma = self.sigma_BH(beta)
        epsilon = self.pot_min

        lambda s.t. B2* = B2 / (2/3 pi sigma)**3 = B2_SW*

        Returns
        -------
        sigma, epsilon, lamda
        """

        # set sigma, eps
        sig = self.sigma_BH(beta)
        eps = self.pot_min

        # B2*
        B2 = self.B2(beta, UB, LB, quad_kwargs=quad_kwargs, **kwargs)
        B2s = B2 / (2./3. * np.pi * sig**3)


        # B2s_SW = 1 + (1-exp(-beta epsilon)) * (lambda**3 - 1)
        #        = B2s
        lam = ((B2s - 1.0)/(1.0 - np.exp(-beta * eps)) + 1.0)**(1.0/3.0)

        return sig, eps, lam


def B2_squarewell(beta, sig, eps, lam):

    return 2./3. * np.pi * sig**3 *(1.0 + (1.0 - np.exp(-beta*eps))*(lam**3 - 1.0))



class LJ_table(object):

    def __init__(self, sig=1.0, eps=1.0, rcut=2.5, forceshift=False, max_fac=1000, beta=1.0, dig=3, rmin=None):

        self.sig = sig
        self.eps = eps
        self.rcut = rcut
        self.forceshift = forceshift
        self.max_fac = max_fac
        self.beta = beta
        self.vmax = self.max_fac / self.beta

        self.dig = dig

        if rmin is not None:
            self._rmin = rmin

    def pot(self, r):
        return phi_lennardjones(r, sig=self.sig, eps=self.eps, rcut=self.rcut, forceshift=self.forceshift)

    @property
    def rmin(self):
        if not hasattr(self, '_rmin'):
            def _f(r):
                return self.pot(r) - self.vmax

            self._rmin = fsolve(_f, 0.1 * self.sig)[0]
        return self._rmin #int(self._rmin * 10**self.dig) / float(10**self.dig)

    @property
    def rrmin(self):
        return int(self.rmin**2 * 10**self.dig) / float(10**self.dig)

    @property
    def rrmax(self):
        return self.rcut**2

    def to_table(self, ds=0.005):

        delta = self.rrmax - self.rrmin
        size = int(delta / ds)
        ds = delta / size

        table = [
            '#BEGIN:HEADER',
            '#DIM',
            '1',
            '#SMIN',
            str(self.rrmin),
            '#SMAX',
            str(self.rrmax),
            '#DS',
            str(ds),
            '#SIZE',
            str(size),
            '#END:HEADER',
            '',
            '#BEGIN:TABLE'
        ]

        for i in range(size):
            rr = self.rrmin + i * ds
            r = np.sqrt(rr)
            table.append('{:.10e} {:.10e}'.format(rr, self.pot(r)))
        table.append('END:TABLE')
        return table



@click.command()
@click.option('--rmin', default=None, type=float, help='optional rmin')
@click.option('--rmax', default=2.5, help='cutoff', type=float)
@click.option('--sig', default=1.0, help='sigma', type=float)
@click.option('--eps', default=1.0, help='epsilon', type=float)
@click.option('--forceshift/--no-forceshift', default=True)
@click.option('--temp',default=1.0, help='temperature', type=float)
@click.option('--ds', default=0.005, help='rr spacing', type=float)
@click.option('--max-fac', default=1000.0, help='temp * max_fac = max p.e.', type=float)
@click.option('--digits', default=3, help='rounding for rrmin', type=int)
@click.option('-o','--output', default=None, help='output file')
def _main(rmin, rmax, sig, eps, forceshift, temp, ds, max_fac, digits, output):
    print('# rmin:',rmin)
    print('# rmax:',rmax)
    print('# sig:',sig)
    print('# eps:',eps)
    print('# forceshift:',forceshift)
    print('# temp:',temp)
    print('# ds input:',ds)
    print('# max_fac:', max_fac)
    print('# digits:',digits)
    print('# output:',output)
    beta = 1.0 / temp
    p = LJ_table(sig=sig, eps=eps, rcut=rmax, forceshift=forceshift,
                 max_fac=max_fac, beta=beta, dig=digits, rmin=rmin)

    print('# rmin calc:', p.rmin)
    print('# v(rmin)', p.pot(p.rmin))

    table = '\n'.join(p.to_table(ds=ds))

    if output is not None:
        with open(output,'w') as f:
            f.write(table)
    else:
        print(table)


# @click.command()
# @click.option('--rmin', default=None, type=float, help='optional rmin')
# @click.option('--rmax', default=2.5, help='cutoff', type=float)
# @click.option('--sig', default=1.0, help='sigma', type=float)
# @click.option('--eps', default=1.0, help='epsilon', type=float)
# @click.option('--forceshift/--no-forceshift', default=True, type=float)
# @click.option('--temp',default=1.0, help='temperature', type=float)
# @click.option('--ds', default=0.005, help='rr spacing', type=float)
# @click.option('--max-fac', default=1000.0, help='temp * max_fac = max p.e.', type=float)
# @click.option('--digits', default=3, help='rounding for rrmin', type=int)
# @click.option('-o','--output', default=None, help='output file')
# def _main(rmin, rmax, sig, eps, forceshift, temp, ds, max_fac):#, digits):#, output):
#     beta = 1.0 / temp
#     p = LJ_table(sig=sig, eps=eps, rcut=rcut, forceshift=forceshift,
#                  max_fac=max_fac, beta=beta, dig=digits, rmin=rmin)

#     table = '\n'.join(p.to_table(ds=ds))

#     if output is not None:
#         with open(output,'w') as f:
#             f.write(table)
#     else:
#         print(table
        # )


if __name__ == '__main__':
    _main()













