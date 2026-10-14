"""Explicit input-route matching; not verification or access authority."""


def require_backend(config, evidence):
    expected = config.get('expected_independent_backend')
    if expected is not None and evidence.get('verification_backend') != expected:
        raise ValueError('verification backend differs from explicit sealed route')
