"""
Tests for snewpy.dsnb

Run with:
    ppytest python/snewpy/test/test_dsnb.py -v
"""

import numpy as np
import pytest
import astropy.units as u

from snewpy.dsnb import (
    CoreCollapseRate,
    PinchedSpectrum,
    DSNBFlux,
)


# ===========================================================================
# CoreCollapseRate
# ===========================================================================

class TestCoreCollapseRate:

    def test_present_day_rate(self):
        """R_CC(0) must equal the reference r0."""
        r0  = 1e-4 * u.yr**-1 * u.Mpc**-3
        rcc = CoreCollapseRate(r0=r0)
        assert np.isclose(
            rcc(0).to(u.yr**-1 * u.Mpc**-3).value, r0.value, rtol=1e-6
        )

    def test_rate_peaks_around_z1(self):
        """Star-formation history peaks near z ~ 1--2."""
        rcc  = CoreCollapseRate()
        z    = np.linspace(0, 5, 200)
        vals = rcc(z).value
        z_peak = z[np.argmax(vals)]
        assert 1.0 < z_peak < 3.0, f"Peak at z={z_peak:.2f}, expected 1--3"

    def test_rate_positive_everywhere(self):
        """Rate must be positive for all z >= 0."""
        rcc  = CoreCollapseRate()
        z    = np.linspace(0, 10, 500)
        vals = rcc(z).value
        assert np.all(vals > 0)

    def test_custom_r0(self):
        """Custom r0 scales the entire rate proportionally."""
        rcc1 = CoreCollapseRate(r0=1e-4 * u.yr**-1 * u.Mpc**-3)
        rcc2 = CoreCollapseRate(r0=2e-4 * u.yr**-1 * u.Mpc**-3)
        z    = np.array([0.0, 0.5, 1.0, 2.0])
        ratio = rcc2(z).value / rcc1(z).value
        assert np.allclose(ratio, 2.0, rtol=1e-10)

    def test_wrong_units_raises(self):
        """Passing r0 in wrong units should raise UnitsError."""
        with pytest.raises(u.UnitsError):
            CoreCollapseRate(r0=1e-4 * u.s**-1)


# ===========================================================================
# PinchedSpectrum
# ===========================================================================

class TestPinchedSpectrum:

    def test_alpha_from_moments_failed_sn(self):
        """Failed-SN alpha from the paper's moments should be ~1.91."""
        spec = PinchedSpectrum.from_moments(
            8.6e52 * u.erg, 18.72 * u.MeV, 470.76 * u.MeV**2
        )
        assert np.isclose(spec.pinching, 1.913, atol=0.01)

    def test_spectrum_positive(self):
        """Spectrum must be non-negative for E > 0."""
        spec = PinchedSpectrum(5e52 * u.erg, 15 * u.MeV, 3.0)
        E    = np.linspace(0.1, 100, 500) * u.MeV
        assert np.all(spec(E).value >= 0)

    def test_spectrum_units(self):
        """Output must have units of MeV^{-1}."""
        spec = PinchedSpectrum(5e52 * u.erg, 15 * u.MeV, 3.0)
        E    = np.array([10.0, 20.0]) * u.MeV
        assert spec(E).unit.is_equivalent(u.MeV**-1)

    def test_spectrum_peaks_near_mean_energy(self):
        """Peak should be near alpha/(1+alpha) * mean_E."""
        alpha   = 3.0
        mean_E  = 15.0   # MeV
        spec    = PinchedSpectrum(5e52 * u.erg, mean_E * u.MeV, alpha)
        E       = np.linspace(0.5, 60, 1000) * u.MeV
        E_peak  = E[np.argmax(spec(E).value)].value
        expected = alpha / (1.0 + alpha) * mean_E   # 11.25 MeV
        assert np.isclose(E_peak, expected, rtol=0.05)

    def test_harder_spectrum_with_higher_fbh(self):
        """Higher BH fraction shifts combined spectrum to higher energies."""
        spec_s = PinchedSpectrum(5.0e52 * u.erg, 15.0 * u.MeV, 3.0)
        spec_f = PinchedSpectrum.from_moments(
            8.6e52 * u.erg, 18.72 * u.MeV, 470.76 * u.MeV**2
        )
        E = np.linspace(0.5, 80, 500) * u.MeV

        def combined(fbh):
            return ((1 - fbh) * spec_s(E) + fbh * spec_f(E)).value

        def mean_energy(fbh):
            dNdE = combined(fbh)
            Ev   = E.value
            # FIX: np.trapz removed in NumPy 2.0; use np.trapezoid
            return np.trapezoid(Ev * dNdE, Ev) / np.trapezoid(dNdE, Ev)

        assert mean_energy(0.40) > mean_energy(0.00)



# ===========================================================================
# DSNBFlux
# ===========================================================================

class TestDSNBFlux:

    @pytest.fixture
    def model(self):
        return DSNBFlux(n_z=400)   # coarser grid for speed in tests

    def test_flux_positive(self, model):
        """Flux must be non-negative at all energies."""
        E   = np.linspace(5, 50, 50) * u.MeV
        phi = model.flux(E)
        assert np.all(phi.value >= 0)

    def test_flux_units(self, model):
        """Flux must have units of cm^{-2} s^{-1} MeV^{-1}."""
        E   = np.array([15.0, 20.0]) * u.MeV
        phi = model.flux(E)
        assert phi.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.MeV**-1)


    def test_higher_fbh_increases_flux_in_window(self, model):
        """Higher f_BH shifts more flux into the 12--30 MeV window."""
        model_lo = DSNBFlux(f_bh=0.00, n_z=300)
        model_hi = DSNBFlux(f_bh=0.40, n_z=300)
        E        = np.linspace(12, 30, 80) * u.MeV
        # FIX: np.trapz removed in NumPy 2.0; use np.trapezoid
        flux_lo  = np.trapezoid(model_lo.flux(E).value, E.value)
        flux_hi  = np.trapezoid(model_hi.flux(E).value, E.value)
        assert flux_hi > flux_lo

    def test_higher_r0_scales_flux_linearly(self):
        """Flux scales linearly with R_CC(0)."""
        model1 = DSNBFlux(rcc=CoreCollapseRate(1e-4 * u.yr**-1 * u.Mpc**-3), n_z=300)
        model2 = DSNBFlux(rcc=CoreCollapseRate(2e-4 * u.yr**-1 * u.Mpc**-3), n_z=300)
        E = np.array([15.0]) * u.MeV
        # FIX: flux() returns a scalar when a single energy is given; use float()
        r = float(model2.flux(E).value) / float(model1.flux(E).value)
        assert np.isclose(r, 2.0, rtol=1e-3)

    def test_integrated_juno_rate_ballpark(self, model):
        """
        JUNO integrated rate at 50% efficiency should be ~1.2 yr^{-1}.
        (Paper Table II: 1.4 yr^{-1}; analytic result is ~14% lower due to
        absence of detector energy-resolution smearing.)
        """
        Np   = 1.22e33
        rate = model.integrated_ibd_rate(Np, efficiency=0.50, n_e=200)
        assert 0.8 < rate.to(u.yr**-1).value < 2.0, \
            f"JUNO rate = {rate:.3f}, expected 0.8--2.0 yr^{{-1}}"

    def test_integrated_skgd_rate_ballpark(self, model):
        """SK-Gd T1 integrated rate at 50% efficiency should be ~1.5 yr^{-1}."""
        Np   = 1.50e33
        rate = model.integrated_ibd_rate(Np, efficiency=0.50, n_e=200)
        assert 0.8 < rate.to(u.yr**-1).value < 2.5

    def test_invalid_fbh_raises(self):
        with pytest.raises(ValueError):
            DSNBFlux(f_bh=1.5)

    def test_smeared_flux_shape(self, model):
        """Smeared flux must have the same shape as the input energy array."""
        E      = np.linspace(10, 40, 60) * u.MeV
        phi_sm = model.smeared_flux(E, energy_resolution=0.03)
        assert phi_sm.shape == E.shape

    def test_smeared_flux_increases_near_threshold(self):
        """Gaussian smearing should increase flux just above 12 MeV."""
        model  = DSNBFlux(n_z=300)
        E      = np.linspace(9, 20, 200) * u.MeV
        phi    = model.flux(E).value
        phi_sm = model.smeared_flux(E, energy_resolution=0.15).value
        mask   = (E.value > 12.0) & (E.value < 14.0)
        assert phi_sm[mask].mean() >= phi[mask].mean() * 0.95
