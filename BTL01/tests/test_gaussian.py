import numpy as np
from scipy.stats import multivariate_normal

from src.data.gaussian import C, MU, NORM, gaussian_pdf, sample_uniform


def test_gaussian_pdf_matches_scipy_and_peak_value():
    points = np.array([[0.0, 0.0], [1.0, -2.0], [-1.5, 3.0]])

    actual = gaussian_pdf(points)
    expected = multivariate_normal(mean=MU, cov=C).pdf(points)

    assert actual.shape == (len(points),)
    assert np.allclose(actual, expected)
    assert np.isclose(actual[0], NORM, atol=1e-12)
    assert np.all(actual >= 0.0)


def test_sample_uniform_is_reproducible_and_in_domain():
    first = sample_uniform(64, np.random.default_rng(2026))
    second = sample_uniform(64, np.random.default_rng(2026))

    assert first.shape == (64, 2)
    assert np.array_equal(first, second)
    assert np.all((-2.0 <= first[:, 0]) & (first[:, 0] <= 2.0))
    assert np.all((-4.0 <= first[:, 1]) & (first[:, 1] <= 4.0))
