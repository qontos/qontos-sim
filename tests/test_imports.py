"""Smoke tests: verify that all top-level packages are importable."""


def test_import_qontos_sim():
    import qontos_sim  # noqa: F401


def test_import_qontos_tensor():
    import qontos_tensor  # noqa: F401


def test_import_qontos_twin():
    import qontos_twin  # noqa: F401
