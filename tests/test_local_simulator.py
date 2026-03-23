"""Basic tests for the local simulator executor."""

import pytest


def test_local_simulator_executor_exists():
    """Verify LocalSimulatorExecutor can be imported."""
    from qontos_sim.local import LocalSimulatorExecutor

    assert LocalSimulatorExecutor is not None


def test_local_simulator_executor_instantiation():
    """Attempt to create a LocalSimulatorExecutor instance."""
    from qontos_sim.local import LocalSimulatorExecutor
    import inspect

    sig = inspect.signature(LocalSimulatorExecutor)
    params = sig.parameters

    # If the constructor takes no required args (besides self), instantiate it
    required = [
        p
        for p in params.values()
        if p.default is inspect.Parameter.empty and p.name != "self"
    ]
    if not required:
        instance = LocalSimulatorExecutor()
        assert instance is not None
    else:
        pytest.skip(
            f"LocalSimulatorExecutor requires args: {[p.name for p in required]}"
        )
