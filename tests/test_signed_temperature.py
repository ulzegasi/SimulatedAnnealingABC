"""Signed-temperature inversion and acceptance regression tests."""

import importlib
import math

import numpy as np
import pytest
from scipy.integrate import quad

from simulated_annealing_abc.sabc import (
    _inverse_mean_energy,
    _update_batch,
    update_epsilon_multi_eps,
)


@pytest.mark.parametrize("beta", [-100., -10., -1., -.1, -.001, -1e-8,
                                  0., 1e-8, .001, .1, 1., 10., 100.])
def test_inverse_against_quadrature(beta: float) -> None:
    """Invert independently integrated normalized bounded densities."""
    normalizer = quad(lambda x: math.exp(-beta * x), 0., 1.)[0]
    mean = quad(lambda x: x * math.exp(-beta * x), 0., 1.)[0] / normalizer
    assert _inverse_mean_energy(mean) == pytest.approx(beta, rel=1e-10, abs=2e-13)


@pytest.mark.parametrize("u", [0., 1e-300, 1e-12, .1, .499999999999, .5])
def test_inverse_reflection_and_endpoints(u: float) -> None:
    """Handle both endpoints and the zero-beta crossing without overflow."""
    beta = _inverse_mean_energy(u)
    reflected = _inverse_mean_energy(1. - u)
    assert math.isfinite(beta) and math.isfinite(reflected)
    assert beta == pytest.approx(-reflected, rel=3e-5, abs=1e-14)


@pytest.mark.parametrize("u", [-.1, 1.1, math.nan, math.inf])
def test_inverse_invalid_energy(u: float) -> None:
    """Do not silently clip invalid CDF outputs."""
    with pytest.raises(ValueError, match="Mean transformed energy"):
        _inverse_mean_energy(u)


def test_inverse_one_ulp_from_half() -> None:
    """Do not let an absolute root tolerance swallow the zero crossing."""
    for u in (np.nextafter(.5, 0.), np.nextafter(.5, 1.)):
        assert _inverse_mean_energy(float(u)) == pytest.approx(12 * (.5-u), rel=1e-14, abs=0.)


def test_signed_curved_force_formula() -> None:
    """Recover signed internal betas without clipping means above one-half."""
    beta = np.array([-4., -2., 0., 2., 4.])
    means = np.array([.5 if b == 0 else 1 / b - 1 / np.expm1(b) for b in beta])
    n = len(beta)
    delta = np.log(means) - np.log(means).mean()
    rho = np.sqrt(n / (4 * (n + 1)) * (delta @ delta))
    cn = math.factorial(2*n+2)/(math.factorial(n+1)*math.factorial(n+2))
    force = 1 / (cn * means * np.sqrt(np.prod(means))) * (
        np.cos(rho) + n / (2*(n+1)) * np.sinc(rho/np.pi) * delta)
    actual = 1 / update_epsilon_multi_eps(means[None, :], 1.)
    np.testing.assert_allclose(actual, beta + force, rtol=1e-11, atol=1e-13)


def test_exact_external_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero beta is represented by infinite epsilon, with no division error."""
    module = importlib.import_module('simulated_annealing_abc.sabc')
    force = math.exp(-math.log(2.) - math.log(.5) - .5 * np.log(.5))
    monkeypatch.setattr(module, '_inverse_mean_energy', lambda u: -force)
    with np.errstate(all='raise'):
        epsilon = update_epsilon_multi_eps(np.array([[.5]]), 1.)
        assert np.isposinf(epsilon[0])
        assert (1 / epsilon)[0] == 0.


@pytest.mark.parametrize("beta, expected", [(-2., 1), (0., 1), (2., 0)])
def test_acceptance_with_signed_beta(beta: float, expected: int) -> None:
    """The unchanged Metropolis kernel handles negative and zero beta."""
    class Prior:
        def logpdf(self, theta: np.ndarray) -> np.ndarray:
            return np.zeros(len(theta))

    def proposal(theta: np.ndarray, inactive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta + .8, np.zeros(len(theta))

    def copy_values(values: np.ndarray, out: np.ndarray) -> np.ndarray:
        out[:] = values
        return out

    population = np.array([[.1], [.2]])
    accepted = _update_batch(
        population=population, u=population.copy(), rho=population.copy(),
        logprior=np.zeros(2), active=slice(0, 1), pop_inactive=population[1:].copy(),
        proposal=proposal, prior=Prior(), f_dist=copy_values, cdfs_dist_prior=copy_values,
        inv_epsilon=np.array([beta]), rho_buf=np.empty((1, 1)), u_buf=np.empty((1, 1)),
        rng=np.random.default_rng(0),
    )
    assert accepted == expected


@pytest.mark.parametrize("mean", [
    [.0218170448162374, .02883638988229307, .43941781058384743, .3941883856539762,
     .49483545559423825, .46159068204653586, .3944101363311906, .37570733808990076,
     .4293744855526324, .38673992983156524, .5006920586003059],
    [.011755031491833889, .00981427194759695, .36042805200673, .46927118950207797,
     .4985336992928813, .4761578967726881, .2615021548722523, .3757775058990849,
     .3648038134440152, .44887268918196627, .4122241687549939],
    [.021860694959283482, .03314035294903349, .4754055709776444, .47409014620774476,
     .5133078091122495, .4516248689288188, .436648531077196, .40268146573713703,
     .5056260474826395, .4041033717001127, .4549449302627685],
    [.018360674673895808, .018157020765115422, .45497853144282097, .39804651709922645,
     .42865860057528493, .4319239550020782, .35414276630897407, .5020125811730625,
     .3708348219861807, .34381692301247097, .35094382205809016],
])
def test_failed_paper_states(mean: list[float]) -> None:
    """All four recorded distractor failures now admit finite signed betas."""
    epsilon = update_epsilon_multi_eps(np.array([mean]), 1.)
    assert np.all(np.isfinite(1 / epsilon))
    assert np.any(epsilon < 0)
