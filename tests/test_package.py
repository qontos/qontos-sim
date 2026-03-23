"""QGH-3013: Package-level tests for qontos-sim.

Tests verify that all three sub-packages are importable, versions are
accessible, there are no circular imports, and __all__ exports are correct.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ===================================================================
# Importability
# ===================================================================

class TestPackageImports:
    """All three top-level packages must be importable."""

    def test_qontos_sim_importable(self):
        import qontos_sim
        assert qontos_sim is not None

    def test_qontos_twin_importable(self):
        import qontos_twin
        assert qontos_twin is not None

    def test_qontos_tensor_importable(self):
        import qontos_tensor
        assert qontos_tensor is not None


# ===================================================================
# Version accessible
# ===================================================================

class TestVersionAccess:
    """__version__ must be available on every package."""

    def test_qontos_sim_version(self):
        import qontos_sim
        assert isinstance(qontos_sim.__version__, str)
        assert len(qontos_sim.__version__) > 0

    def test_qontos_twin_version(self):
        import qontos_twin
        assert isinstance(qontos_twin.__version__, str)
        assert len(qontos_twin.__version__) > 0

    def test_qontos_tensor_version(self):
        import qontos_tensor
        assert isinstance(qontos_tensor.__version__, str)
        assert len(qontos_tensor.__version__) > 0

    def test_versions_match(self):
        import qontos_sim
        import qontos_twin
        import qontos_tensor

        assert qontos_sim.__version__ == qontos_twin.__version__
        assert qontos_twin.__version__ == qontos_tensor.__version__


# ===================================================================
# No circular imports
# ===================================================================

class TestNoCircularImports:
    """Reimporting after cache clear should succeed without cycles."""

    def test_reimport_qontos_sim(self):
        """Force reimport of qontos_sim to detect cycles."""
        mod_keys = [k for k in sys.modules if k.startswith("qontos_sim")]
        saved = {k: sys.modules.pop(k) for k in mod_keys}
        try:
            import qontos_sim  # fresh import
            assert qontos_sim.__version__
        finally:
            # Restore to avoid side effects on other tests
            sys.modules.update(saved)

    def test_reimport_qontos_twin(self):
        mod_keys = [k for k in sys.modules if k.startswith("qontos_twin")]
        saved = {k: sys.modules.pop(k) for k in mod_keys}
        try:
            import qontos_twin
            assert qontos_twin.__version__
        finally:
            sys.modules.update(saved)

    def test_reimport_qontos_tensor(self):
        mod_keys = [k for k in sys.modules if k.startswith("qontos_tensor")]
        saved = {k: sys.modules.pop(k) for k in mod_keys}
        try:
            import qontos_tensor
            assert qontos_tensor.__version__
        finally:
            sys.modules.update(saved)


# ===================================================================
# __all__ exports are correct
# ===================================================================

class TestAllExports:
    """Every name in __all__ must be resolvable on the module."""

    def test_qontos_sim_all(self):
        import qontos_sim

        for name in qontos_sim.__all__:
            obj = getattr(qontos_sim, name, None)
            assert obj is not None, f"qontos_sim.__all__ lists '{name}' but it is not accessible"

    def test_qontos_twin_all(self):
        import qontos_twin

        for name in qontos_twin.__all__:
            obj = getattr(qontos_twin, name, None)
            assert obj is not None, f"qontos_twin.__all__ lists '{name}' but it is not accessible"

    def test_qontos_tensor_all(self):
        import qontos_tensor

        for name in qontos_tensor.__all__:
            obj = getattr(qontos_tensor, name, None)
            assert obj is not None, f"qontos_tensor.__all__ lists '{name}' but it is not accessible"

    def test_qontos_sim_all_contains_key_exports(self):
        import qontos_sim

        expected = {
            "__version__",
            "LocalSimulatorExecutor",
            "NoisySimulatorExecutor",
            "ValidationResult",
            "aer_result_to_partition_result",
        }
        assert expected.issubset(set(qontos_sim.__all__))

    def test_qontos_tensor_all_contains_key_exports(self):
        import qontos_tensor

        expected = {
            "__version__",
            "__stability__",
            "Tensor",
            "TensorNetwork",
            "MatrixProductState",
            "MatrixProductOperator",
            "DMRG",
            "DMRGConfig",
            "DMRGResult",
            "TNSimulator",
            "GateInstruction",
            "SimulationResult",
        }
        assert expected.issubset(set(qontos_tensor.__all__))
