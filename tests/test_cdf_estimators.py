"""Unit tests for CDF estimators."""

import numpy as np
import pytest

from simulated_annealing_abc.cdf_estimators import CDF1D, CDFMulti, build_cdf


class TestBuildCDF:
    """Tests for build_cdf factory function."""

    def test_build_cdf_1d(self):
        """Test build_cdf with 1D input returns CDF1D."""
        x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        cdf = build_cdf(x)

        assert isinstance(cdf, CDF1D)
        assert cdf.values is not None
        assert cdf.probs is not None

    def test_build_cdf_2d(self):
        """Test build_cdf with 2D input returns CDFMulti."""
        x = np.array(
            [
                [0.5, 1.0],
                [1.0, 2.0],
                [1.5, 3.0],
                [2.0, 4.0],
            ]
        )
        cdf = build_cdf(x)

        assert isinstance(cdf, CDFMulti)
        assert cdf.n_stats == 2
        assert len(cdf.values_list) == 2
        assert len(cdf.probs_list) == 2

    def test_build_cdf_1d_shape(self):
        """Test that CDF1D has expected shape."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cdf = build_cdf(x)

        assert cdf.values.ndim == 1
        assert cdf.probs.ndim == 1
        assert len(cdf.values) == len(cdf.probs)

    def test_build_cdf_2d_shape(self):
        """Test that CDFMulti has expected shapes."""
        x = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        cdf = build_cdf(x)

        for values, probs in zip(cdf.values_list, cdf.probs_list):
            assert values.ndim == 1
            assert probs.ndim == 1
            assert len(values) == len(probs)

    def test_build_cdf_invalid_dim(self):
        """Test that 3D input raises error."""
        x = np.ones((2, 2, 2))

        with pytest.raises(ValueError, match="1D or 2D"):
            build_cdf(x)

    def test_build_cdf_negative_values(self):
        """Test that negative distances raise error."""
        x = np.array([-1.0, 1.0, 2.0])

        with pytest.raises(ValueError, match="non-negative"):
            build_cdf(x)

    def test_build_cdf_infinite_values(self):
        """Test that infinite values raise error."""
        x = np.array([1.0, np.inf, 2.0])

        with pytest.raises(ValueError, match="finite"):
            build_cdf(x)

    def test_build_cdf_all_zeros(self):
        """Test that all-zero input raises error."""
        x = np.zeros(10)

        with pytest.raises(ValueError, match="all.*zero"):
            build_cdf(x)


class TestCDF1D:
    """Tests for CDF1D class."""

    def test_call_scalar(self):
        """Test CDF1D with scalar input."""
        values = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([0.0, 0.33, 0.67, 1.0])
        cdf = CDF1D(values=values, probs=probs)

        result = cdf(1.5)

        assert np.isscalar(result) or result.shape == ()
        assert 0.0 <= result <= 1.0

    def test_call_1d_array(self):
        """Test CDF1D with 1D array input."""
        values = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([0.0, 0.33, 0.67, 1.0])
        cdf = CDF1D(values=values, probs=probs)

        d = np.array([0.5, 1.5, 2.5])
        result = cdf(d)

        assert result.shape == (3,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_call_2d_array(self):
        """Test CDF1D with 2D array input."""
        values = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([0.0, 0.33, 0.67, 1.0])
        cdf = CDF1D(values=values, probs=probs)

        d = np.array([[0.5, 1.5], [2.5, 3.5]])
        result = cdf(d)

        assert result.shape == d.shape

    def test_call_with_out_buffer(self):
        """Test CDF1D with pre-allocated output buffer."""
        values = np.array([0.0, 1.0, 2.0, 3.0])
        probs = np.array([0.0, 0.33, 0.67, 1.0])
        cdf = CDF1D(values=values, probs=probs)

        d = np.array([0.5, 1.5, 2.5])
        out = np.empty(3)
        result = cdf(d, out=out)

        assert result is out

    def test_interpolation_left(self):
        """Test that values below min return 0."""
        values = np.array([0.0, 1.0, 2.0])
        probs = np.array([0.0, 0.5, 1.0])
        cdf = CDF1D(values=values, probs=probs)

        result = cdf(-1.0)

        assert result == 0.0

    def test_interpolation_right(self):
        """Test that values above max return 1."""
        values = np.array([0.0, 1.0, 2.0])
        probs = np.array([0.0, 0.5, 1.0])
        cdf = CDF1D(values=values, probs=probs)

        result = cdf(10.0)

        assert result == 1.0

    def test_monotonicity(self):
        """Test that CDF is monotonic."""
        x = np.abs(np.random.default_rng(42).standard_normal(100))
        cdf = build_cdf(x)

        d_sorted = np.sort(np.abs(np.random.default_rng(43).standard_normal(50)))
        results = cdf(d_sorted)

        assert np.all(np.diff(results) >= -1e-10)


class TestCDFMulti:
    """Tests for CDFMulti class."""

    def test_call_1d_array(self):
        """Test CDFMulti with single row (1D after squeeze)."""
        n_stats = 2
        values_list = [
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 10.0, 20.0, 30.0]),
        ]
        probs_list = [
            np.array([0.0, 0.33, 0.67, 1.0]),
            np.array([0.0, 0.33, 0.67, 1.0]),
        ]
        cdf = CDFMulti(values_list=values_list, probs_list=probs_list, n_stats=n_stats)

        rho = np.array([1.5, 15.0])
        result = cdf(rho)

        assert result.shape == (n_stats,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_call_2d_array(self):
        """Test CDFMulti with batch input."""
        n_stats = 2
        values_list = [
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 10.0, 20.0, 30.0]),
        ]
        probs_list = [
            np.array([0.0, 0.33, 0.67, 1.0]),
            np.array([0.0, 0.33, 0.67, 1.0]),
        ]
        cdf = CDFMulti(values_list=values_list, probs_list=probs_list, n_stats=n_stats)

        rho = np.array(
            [
                [1.5, 15.0],
                [2.5, 25.0],
                [0.5, 5.0],
            ]
        )
        result = cdf(rho)

        assert result.shape == (3, n_stats)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_call_with_out_buffer(self):
        """Test CDFMulti with pre-allocated output buffer."""
        n_stats = 2
        values_list = [
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 10.0, 20.0, 30.0]),
        ]
        probs_list = [
            np.array([0.0, 0.33, 0.67, 1.0]),
            np.array([0.0, 0.33, 0.67, 1.0]),
        ]
        cdf = CDFMulti(values_list=values_list, probs_list=probs_list, n_stats=n_stats)

        rho = np.array([[1.5, 15.0], [2.5, 25.0]])
        out = np.empty((2, n_stats))
        result = cdf(rho, out=out)

        assert result is out

    def test_call_out_buffer_wrong_shape(self):
        """Test that wrong-shaped buffer raises error."""
        n_stats = 2
        values_list = [
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 10.0, 20.0]),
        ]
        probs_list = [
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        ]
        cdf = CDFMulti(values_list=values_list, probs_list=probs_list, n_stats=n_stats)

        rho = np.array([[1.5, 15.0]])
        out = np.empty((5, n_stats))

        with pytest.raises(ValueError, match="shape"):
            cdf(rho, out=out)

    def test_wrong_n_stats(self):
        """Test that wrong number of statistics raises error."""
        n_stats = 2
        values_list = [
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 10.0, 20.0]),
        ]
        probs_list = [
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
        ]
        cdf = CDFMulti(values_list=values_list, probs_list=probs_list, n_stats=n_stats)

        rho = np.array([1.5, 15.0, 25.0])

        with pytest.raises(ValueError, match="Expected rho with 2 columns"):
            cdf(rho)

    def test_per_column_interpolation(self):
        """Test that each column is interpolated independently."""
        x = np.array(
            [
                [1.0, 100.0],
                [2.0, 200.0],
                [3.0, 300.0],
                [4.0, 400.0],
            ]
        )
        cdf = build_cdf(x)

        rho = np.array([[2.5, 250.0]])
        result = cdf(rho)

        assert result.shape == (1, 2)
        assert result[0, 0] >= 0.5
        assert result[0, 1] >= 0.5


class TestCDFEdgeCases:
    """Edge case tests for CDF estimators."""

    def test_single_value(self):
        """Test CDF with single unique value (after adding 0 and inflated max)."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        cdf = build_cdf(x)

        assert isinstance(cdf, CDF1D)

        d = np.array([0.5, 1.0, 1.5, 2.0])
        result = cdf(d)

        assert result[0] >= 0.0
        assert result[-1] == 1.0

    def test_many_repeated_values(self):
        """Test CDF with many repeated values."""
        x = np.array([1.0] * 50 + [2.0] * 50)
        cdf = build_cdf(x)

        assert isinstance(cdf, CDF1D)

        result = cdf(1.5)
        assert 0.0 < result < 1.0

    def test_very_small_values(self):
        """Test CDF with very small positive values."""
        x = np.array([1e-10, 1e-9, 1e-8, 1e-7])
        cdf = build_cdf(x)

        assert isinstance(cdf, CDF1D)

        result = cdf(5e-9)
        assert 0.0 < result < 1.0

    def test_large_values(self):
        """Test CDF with large values."""
        x = np.array([1e6, 1e7, 1e8, 1e9])
        cdf = build_cdf(x)

        assert isinstance(cdf, CDF1D)

        result = cdf(5e7)
        assert 0.0 < result < 1.0

    def test_inflation_factor(self):
        """Test that inflation factor 'a' extends the max value."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        cdf_default = build_cdf(x, a=1.5)
        cdf_large = build_cdf(x, a=2.0)

        assert cdf_default.values[-1] == 5.0 * 1.5
        assert cdf_large.values[-1] == 5.0 * 2.0
