"""
Diffuse Supernova Neutrino Background (DSNB) model for SNEWPY.

Implements the DSNB as a steady-state neutrino source analogous to
SNEWPY's existing supernova model classes.  Use get_flux() to compute
the DSNB flux, integrate over the detector exposure, and pass to
RateCalculator.

Physics follows Li, Vagins & Wurm (2022) [arXiv:2201.12920].
Core-collapse rate: Hopkins & Beacom (2006), ApJ 651, 142.
Cosmology: Planck 2018.

Example
-------
>>> from snewpy.models.ccsn import Nakazato_2013
>>> from snewpy.dsnb import DSNB
>>> from snewpy.flavor_transformation import NoTransformation
>>> from snewpy.rate_calculator import RateCalculator
>>> import astropy.units as u
>>> import numpy as np
>>>
>>> sn_models = [
...     (Nakazato_2013(progenitor_mass=13*u.Msun, revival_time=100*u.ms,
...                    metallicity=0.02, eos='shen'), 13),
...     (Nakazato_2013(progenitor_mass=20*u.Msun, revival_time=100*u.ms,
...                    metallicity=0.02, eos='shen'), 20),
... ]
>>> model    = DSNB(sn_models)
>>> times    = [0, 1] * u.yr
>>> energies = np.linspace(0, 50, 201) * u.MeV
>>> flux     = model.get_flux(t=times, E=energies,
...                           flavor_xform=NoTransformation())
>>> fluence  = flux.integrate('time')
>>> rc       = RateCalculator()
>>> events   = rc.run(fluence, "wc100kt30prct")
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
from snewpy.neutrino import Flavor
from snewpy.flux import Flux, Axes
from typing import Optional, List, Tuple

__all__ = ["CoreCollapseRate", "salpeter_imf", "DSNB"]

_ERG_TO_MEV = 6.241509074e5
_MPC_TO_CM  = 3.0856775815e24
_YR_TO_S    = 3.15576e7
_PLANCK18   = FlatLambdaCDM(H0=67.4, Om0=0.315)


class CoreCollapseRate:
    """
    Redshift-dependent core-collapse supernova rate.

    Implements the parametrisation of Hopkins & Beacom (2006),
    ApJ 651, 142, as used in Eq. (2) of Li, Vagins & Wurm (2022).

    Parameters
    ----------
    r0 : astropy.units.Quantity
        Present-day rate.  Default: 1e-4 yr^{-1} Mpc^{-3}.

    Examples
    --------
    >>> import astropy.units as u
    >>> rcc = CoreCollapseRate()
    >>> rcc(0)
    <Quantity 1.e-4 1 / (Mpc3 yr)>
    >>> rcc(1) > rcc(0)
    True
    """

    _a, _b, _c, _d, _h = 0.0170, 0.13, 3.3, 5.3, 0.7

    def __init__(self, r0: u.Quantity = 1e-4 * u.yr**-1 * u.Mpc**-3):
        if not r0.unit.is_equivalent(u.yr**-1 * u.Mpc**-3):
            raise u.UnitsError(
                f"r0 must have units yr^-1 Mpc^-3, got {r0.unit}"
            )
        self.r0 = r0.to(u.yr**-1 * u.Mpc**-3)

    def __call__(self, z) -> u.Quantity:
        """
        Evaluate R_CC(z).

        Parameters
        ----------
        z : float or array_like
            Redshift.

        Returns
        -------
        astropy.units.Quantity
            Rate in yr^{-1} Mpc^{-3}.
        """
        z   = np.asarray(z, dtype=float)
        num = (self._a + self._b * z) ** self._h
        den = self._a**self._h * (1.0 + (z / self._c)**self._d)
        return self.r0 * num / den


def salpeter_imf(masses: np.ndarray) -> np.ndarray:
    """
    Salpeter (1955) initial mass function weights.

    Returns weights proportional to M^{-2.35}.

    Parameters
    ----------
    masses : array_like
        Progenitor masses in solar masses.

    Returns
    -------
    numpy.ndarray
        Unnormalised weights.
    """
    return np.asarray(masses, dtype=float) ** (-2.35)


class DSNB:
    """
    Diffuse Supernova Neutrino Background model.

    Analogous to SNEWPY's existing supernova model classes.  The primary
    method is get_flux(t, E, flavor_xform), which returns a SNEWPY Flux
    object.  Calling flux.integrate('time') gives the fluence over the
    detector exposure window, ready for RateCalculator.

    The DSNB flux integral follows Eq. (1) of Li, Vagins & Wurm (2022):

        dPhi/dE_obs = (c/H0) * integral_0^{z_max}
            R_CC(z) / E(z) * dN/dE_emit[(1+z)*E_obs] * (1+z)  dz

    The per-supernova emission spectrum dN/dE is computed by
    time-integrating the luminosity, mean energy, and pinching parameter
    from each SNEWPY model, then taking the Salpeter IMF-weighted average
    over all progenitor masses.

    Parameters
    ----------
    sn_models : list of (snewpy SupernovaModel, float) tuples
        Each tuple is (model_instance, progenitor_mass_in_solar_masses).
        Duplicate masses raise ValueError.
        The IMF weight for each progenitor is the integral of the Salpeter
        (1955) IMF over the mass interval it represents, using midpoints
        between adjacent progenitor masses as interval boundaries, with
        8 and 100 Msun as the outer limits.
    rcc : CoreCollapseRate, optional
        Redshift-dependent core-collapse rate.
        Default: Hopkins & Beacom (2006) with R_CC(0) = 1e-4 yr^-1 Mpc^-3.
    cosmology : astropy cosmology instance, optional
        Default: Planck 2018 (H0=67.4, Om0=0.315).
    z_max : float, optional
        Upper redshift integration limit.  Default: 5.0.
    n_z : int, optional
        Number of redshift quadrature nodes.  Default: 800.

    Notes
    -----
    The current implementation uses the NU_E_BAR component of each SNEWPY
    model (the IBD target).  The flavor_xform argument is reserved for a
    future version that applies oscillation effects self-consistently
    inside the redshift integral.

    Examples
    --------
    >>> from snewpy.models.ccsn import Nakazato_2013
    >>> from snewpy.dsnb import DSNB
    >>> from snewpy.flavor_transformation import NoTransformation
    >>> from snewpy.rate_calculator import RateCalculator
    >>> import astropy.units as u, numpy as np
    >>>
    >>> sn_models = [
    ...     (Nakazato_2013(progenitor_mass=13*u.Msun,
    ...                    revival_time=100*u.ms,
    ...                    metallicity=0.02, eos='shen'), 13),
    ...     (Nakazato_2013(progenitor_mass=20*u.Msun,
    ...                    revival_time=100*u.ms,
    ...                    metallicity=0.02, eos='shen'), 20),
    ... ]
    >>> model   = DSNB(sn_models)
    >>> flux    = model.get_flux(t=[0,1]*u.yr,
    ...                          E=np.linspace(0,50,201)*u.MeV,
    ...                          flavor_xform=NoTransformation())
    >>> fluence = flux.integrate('time')
    >>> rc      = RateCalculator()
    >>> events  = rc.run(fluence, "wc100kt30prct")
    """

    _E_FINE = np.geomspace(0.1, 300.0, 600) * u.MeV

    def __init__(
        self,
        sn_models : List[Tuple],
        rcc       : Optional[CoreCollapseRate] = None,
        cosmology                              = None,
        z_max     : float                      = 5.0,
        n_z       : int                        = 800,
    ):
        masses = [float(m) for _, m in sn_models]
        if len(masses) != len(set(masses)):
            raise ValueError(
                "Duplicate progenitor masses in sn_models. "
                "Each mass must appear at most once."
            )
        self._sorted  = sorted(sn_models, key=lambda p: p[1])
        self._masses  = np.array([m for _, m in self._sorted], dtype=float)
        self._weights = self._imf_weights(self._masses)
        self.rcc       = rcc or CoreCollapseRate()
        self.cosmology = cosmology or _PLANCK18
        self.z_max     = float(z_max)
        self.n_z       = int(n_z)

    @staticmethod
    def _imf_weights(sorted_masses: np.ndarray) -> np.ndarray:
        """
        Salpeter IMF integral weights for unevenly spaced progenitor masses.

        Weight for progenitor i = integral of M^{-2.35} dM over its mass
        interval, where intervals are defined by midpoints between adjacent
        masses (8 and 100 Msun as outer limits).
        """
        n          = len(sorted_masses)
        edges      = np.empty(n + 1)
        edges[0]   = 8.0
        edges[-1]  = 100.0
        for i in range(1, n):
            edges[i] = 0.5 * (sorted_masses[i - 1] + sorted_masses[i])
        weights = np.array([
            quad(lambda m: m**(-2.35), edges[i], edges[i + 1])[0]
            for i in range(n)
        ])
        return weights / weights.sum()

    def _dNdE_per_sn(self, E_grid: u.Quantity) -> np.ndarray:
        """
        IMF-weighted, time-integrated dN/dE [MeV^{-1}] per core collapse.

        For each SNEWPY model the burst-integrated emission spectrum is
        computed by time-integrating the pinched spectrum constructed from
        the model's luminosity, meanE, and pinch time series for
        Flavor.NU_E_BAR.  Results are averaged with Salpeter IMF weights.
        """
        E_vals     = E_grid.to(u.MeV).value
        dNdE_total = np.zeros(len(E_vals))

        for (model, _mass), weight in zip(self._sorted, self._weights):
            t  = model.time.to(u.s).value
            L  = model.luminosity[Flavor.NU_E_BAR].to(u.erg / u.s).value
            Em = model.meanE[Flavor.NU_E_BAR].to(u.MeV).value
            al = np.asarray(model.pinch[Flavor.NU_E_BAR], dtype=float)

            mask         = (L > 0) & (Em > 0)
            t, L, Em, al = t[mask], L[mask], Em[mask], al[mask]

            prefac = (_ERG_TO_MEV * L / Em**2)[:, np.newaxis]
            norm   = ((1 + al)**(1 + al) / gamma_func(1 + al))[:, np.newaxis]
            x      = E_vals[np.newaxis, :] / Em[:, np.newaxis]
            a2     = al[:, np.newaxis]
            spec   = prefac * norm * x**a2 * np.exp(-(1 + a2) * x)

            dNdE_total += weight * np.trapezoid(spec, t, axis=0)

        return dNdE_total

    def get_flux(
        self,
        t           : u.Quantity,
        E           : u.Quantity,
        flavor_xform = None,
    ) -> Flux:
        """
        Compute the DSNB differential flux.

        The DSNB is a steady-state background; the returned Flux has the
        same value at every time sample point in t.  Call
        flux.integrate('time') to obtain the fluence over [t[0], t[-1]],
        then pass to RateCalculator.

        Parameters
        ----------
        t : astropy.units.Quantity
            Time sample points of the exposure window, e.g. [0, 1]*u.yr.
            Must have at least 2 points for flux.integrate('time') to work.
        E : astropy.units.Quantity
            Observed neutrino energies.
        flavor_xform : optional
            Reserved for future oscillation support.  Currently unused.

        Returns
        -------
        snewpy.flux.Flux
            Shape (N_flavor, N_time, N_energy).
            NU_E_BAR carries the full DSNB flux; all other flavors are zero.
        """
        dNdE_fine = self._dNdE_per_sn(self._E_FINE)

        E_obs  = np.atleast_1d(E.to(u.MeV).value)
        z      = np.linspace(1e-4, self.z_max, self.n_z)
        E_emit = np.outer(E_obs, 1.0 + z)

        dNdE_grid = np.interp(
            E_emit.ravel(),
            self._E_FINE.to(u.MeV).value,
            dNdE_fine,
            left=0.0,
            right=0.0,
        ).reshape(E_emit.shape)

        Hz  = self.cosmology.efunc(z)
        rcc = self.rcc(z).to(u.yr**-1 * u.Mpc**-3).value

        integrand = dNdE_grid * (rcc * (1.0 + z) / Hz)[np.newaxis, :]
        integral  = np.trapezoid(integrand, z, axis=1)

        H0_s      = self.cosmology.H(0).to(u.s**-1).value
        c_cm_s    = const.c.to(u.cm / u.s).value
        rate_conv = 1.0 / (_YR_TO_S * _MPC_TO_CM**3)
        phi       = (c_cm_s / H0_s) * integral * rate_conv

        t_vals     = np.atleast_1d(t.to(u.s).value)
        N_t        = len(t_vals)
        flavors    = list(Flavor)
        N_flav     = len(flavors)
        N_E        = len(E_obs)

        data           = np.zeros((N_flav, N_t, N_E))
        nuebar_idx     = flavors.index(Flavor.NU_E_BAR)
        for i_t in range(N_t):
            data[nuebar_idx, i_t, :] = phi

        return Flux(
            data            = data * u.cm**-2 * u.s**-1 * u.MeV**-1,
            flavor          = flavors,
            time            = t.to(u.s),
            energy          = E.to(u.MeV),
            integrable_axes = {Axes.time, Axes.energy},
        )
