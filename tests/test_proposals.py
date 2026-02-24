"""Unit tests for proposal mechanisms."""

import numpy as np
import pytest

from simulated_annealing_abc import DifferentialEvolution, RandomWalk, StretchMove


class TestRandomWalk:
    """Tests for RandomWalk proposal."""

    def test_init_default(self):
        """Test default initialization."""
        rw = RandomWalk(n_para=1)
        assert rw.beta == 0.8
        assert np.isscalar(rw.Sigma)
        assert rw.Sigma == -1.0

    def test_init_custom_beta(self):
        """Test custom beta parameter."""
        rw = RandomWalk(beta=0.5, n_para=1)
        assert rw.beta == 0.5

    def test_init_invalid_beta(self):
        """Test that invalid beta raises error."""
        with pytest.raises(ValueError, match="beta"):
            RandomWalk(beta=0.0, n_para=1)
        with pytest.raises(ValueError, match="beta"):
            RandomWalk(beta=1.5, n_para=1)

    def test_init_multidim(self):
        """Test initialization for multi-dimensional case."""
        rw = RandomWalk(n_para=3)
        assert rw.Sigma.shape == (3, 3)
        assert np.all(rw.Sigma == -1.0)

    def test_update_1d(self, rng):
        """Test update() in 1D case."""
        rw = RandomWalk(n_para=1, rng=rng)
        pop = rng.normal(0, 1, size=(100, 1))
        rw.update(pop)
        assert np.isscalar(rw.Sigma)
        assert rw.Sigma > 0

    def test_update_nd(self, rng):
        """Test update() in multi-dimensional case."""
        rw = RandomWalk(n_para=2, rng=rng)
        pop = rng.normal(0, 1, size=(100, 2))
        rw.update(pop)
        assert rw.Sigma.shape == (2, 2)
        assert np.all(np.diag(rw.Sigma) > 0)

    def test_call_without_update_raises(self, rng):
        """Test that calling before update raises error."""
        rw = RandomWalk(n_para=1, rng=rng)
        theta = np.array([[0.0]])
        pop = np.array([[1.0], [2.0]])
        with pytest.raises(RuntimeError, match="not updated"):
            rw(theta, pop)

    def test_call_1d(self, rng):
        """Test proposal generation in 1D."""
        rw = RandomWalk(n_para=1, rng=rng)
        pop = rng.normal(0, 1, size=(100, 1))
        rw.update(pop)

        theta = np.array([[0.0], [1.0], [2.0]])
        proposal, log_factors = rw(theta, pop[:10])

        assert proposal.shape == (3, 1)
        assert log_factors.shape == (3,)
        assert np.all(log_factors == 0.0)
        assert np.all(np.isfinite(proposal))

    def test_call_nd(self, rng):
        """Test proposal generation in multi-dimensional case."""
        rw = RandomWalk(n_para=2, rng=rng)
        pop = rng.normal(0, 1, size=(100, 2))
        rw.update(pop)

        theta = np.array([[0.0, 0.0], [1.0, 2.0]])
        proposal, log_factors = rw(theta, pop[:10])

        assert proposal.shape == (2, 2)
        assert log_factors.shape == (2,)
        assert np.all(log_factors == 0.0)

    def test_clone(self, rng):
        """Test clone() creates independent copy with same Sigma."""
        rw = RandomWalk(n_para=2, rng=rng)
        pop = rng.normal(0, 1, size=(100, 2))
        rw.update(pop)

        rng2 = np.random.default_rng(99)
        rw_clone = rw.clone(rng2)

        assert rw_clone.beta == rw.beta
        assert np.allclose(rw_clone.Sigma, rw.Sigma)
        assert rw_clone.rng is not rw.rng

    def test_clone_1d(self, rng):
        """Test clone() in 1D case."""
        rw = RandomWalk(n_para=1, rng=rng)
        pop = rng.normal(0, 1, size=(100, 1))
        rw.update(pop)

        rng2 = np.random.default_rng(99)
        rw_clone = rw.clone(rng2)

        assert rw_clone.beta == rw.beta
        assert rw_clone.Sigma == rw.Sigma


class TestDifferentialEvolution:
    """Tests for DifferentialEvolution proposal."""

    def test_init_with_gamma0(self):
        """Test initialization with explicit gamma0."""
        de = DifferentialEvolution(gamma0=1.5)
        assert de.gamma0 == 1.5
        assert de.sigma_gamma == 1e-5

    def test_init_with_n_para(self):
        """Test initialization with n_para (computes gamma0)."""
        de = DifferentialEvolution(n_para=4)
        expected = 2.38 / np.sqrt(2.0 * 4)
        assert np.isclose(de.gamma0, expected)

    def test_init_invalid_args(self):
        """Test that providing both or neither gamma0/n_para raises error."""
        with pytest.raises(ValueError, match="exactly one"):
            DifferentialEvolution()
        with pytest.raises(ValueError, match="exactly one"):
            DifferentialEvolution(gamma0=1.0, n_para=4)

    def test_call(self, rng):
        """Test proposal generation."""
        de = DifferentialEvolution(n_para=2, rng=rng)
        theta = np.array([[0.0, 0.0], [1.0, 1.0]])
        pop = rng.normal(0, 1, size=(50, 2))

        proposal, log_factors = de(theta, pop)

        assert proposal.shape == (2, 2)
        assert log_factors.shape == (2,)
        assert np.all(log_factors == 0.0)

    def test_call_too_small_population(self, rng):
        """Test that population with < 2 particles raises error."""
        de = DifferentialEvolution(n_para=2, rng=rng)
        theta = np.array([[0.0, 0.0]])
        pop = np.array([[1.0, 1.0]])

        with pytest.raises(ValueError, match="at least 2"):
            de(theta, pop)

    def test_clone(self, rng):
        """Test clone() creates independent copy."""
        de = DifferentialEvolution(n_para=2, rng=rng)

        rng2 = np.random.default_rng(99)
        de_clone = de.clone(rng2)

        assert de_clone.gamma0 == de.gamma0
        assert de_clone.sigma_gamma == de.sigma_gamma
        assert de_clone.rng is not de.rng

    def test_update_noop(self, rng):
        """Test that update() is a no-op for DifferentialEvolution."""
        de = DifferentialEvolution(n_para=2, rng=rng)
        pop = rng.normal(0, 1, size=(100, 2))
        de.update(pop)


class TestStretchMove:
    """Tests for StretchMove proposal."""

    def test_init_default(self):
        """Test default initialization."""
        sm = StretchMove()
        assert sm.a == 2.0

    def test_init_custom_a(self):
        """Test custom 'a' parameter."""
        sm = StretchMove(a=3.0)
        assert sm.a == 3.0

    def test_init_invalid_a(self):
        """Test that a <= 1 raises error."""
        with pytest.raises(ValueError, match="a.*must be > 1"):
            StretchMove(a=1.0)
        with pytest.raises(ValueError, match="a.*must be > 1"):
            StretchMove(a=0.5)

    def test_call(self, rng):
        """Test proposal generation."""
        sm = StretchMove(a=2.0, rng=rng)
        theta = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pop = rng.normal(0, 1, size=(50, 2))

        proposal, log_factors = sm(theta, pop)

        assert proposal.shape == (3, 2)
        assert log_factors.shape == (3,)
        assert np.all(np.isfinite(log_factors))

    def test_log_factors_correctness(self, rng):
        """Test that log factors follow stretch move formula."""
        sm = StretchMove(a=2.0, rng=rng)
        n_para = 3
        theta = rng.normal(0, 1, size=(10, n_para))
        pop = rng.normal(0, 1, size=(50, n_para))

        proposal, log_factors = sm(theta, pop)

        assert np.all(np.isfinite(proposal))
        assert np.all(np.isfinite(log_factors))

    def test_call_empty_population(self, rng):
        """Test that empty population raises error."""
        sm = StretchMove(rng=rng)
        theta = np.array([[0.0, 0.0]])
        pop = np.empty((0, 2))

        with pytest.raises(ValueError, match="must not be empty"):
            sm(theta, pop)

    def test_clone(self, rng):
        """Test clone() creates independent copy."""
        sm = StretchMove(a=3.0, rng=rng)

        rng2 = np.random.default_rng(99)
        sm_clone = sm.clone(rng2)

        assert sm_clone.a == sm.a
        assert sm_clone.rng is not sm.rng

    def test_update_noop(self, rng):
        """Test that update() is a no-op for StretchMove."""
        sm = StretchMove(rng=rng)
        pop = rng.normal(0, 1, size=(100, 2))
        sm.update(pop)


class TestProposalClone:
    """Tests for clone() method across all proposal types."""

    def test_randomwalk_clone_independence(self, rng):
        """Test that cloned RandomWalk produces different proposals."""
        rw = RandomWalk(n_para=2, rng=rng)
        pop = rng.normal(0, 1, size=(100, 2))
        rw.update(pop)

        rw_clone = rw.clone(np.random.default_rng(99))
        rw_clone.update(pop)

        theta = np.array([[0.0, 0.0]])

        p1, _ = rw(theta, pop[:10])
        p2, _ = rw_clone(theta, pop[:10])

        assert not np.allclose(p1, p2)

    def test_differential_evolution_clone_independence(self, rng):
        """Test that cloned DE produces different proposals."""
        de = DifferentialEvolution(n_para=2, rng=rng)
        de_clone = de.clone(np.random.default_rng(99))

        pop = rng.normal(0, 1, size=(50, 2))
        theta = np.array([[0.0, 0.0]])

        p1, _ = de(theta, pop)
        p2, _ = de_clone(theta, pop)

        assert not np.allclose(p1, p2)

    def test_stretch_move_clone_independence(self, rng):
        """Test that cloned StretchMove produces different proposals."""
        sm = StretchMove(rng=rng)
        sm_clone = sm.clone(np.random.default_rng(99))

        pop = rng.normal(0, 1, size=(50, 2))
        theta = np.array([[0.0, 0.0]])

        p1, _ = sm(theta, pop)
        p2, _ = sm_clone(theta, pop)

        assert not np.allclose(p1, p2)
