"""
Basic tests for RiTINI package functionality.
"""

import pytest


def test_import_ritini():
    """Test that the package can be imported."""
    import ritini
    assert ritini.__version__ == "0.1.0"


def test_import_core():
    """Test that core modules can be imported."""
    from ritini.core import utils
    # Just test that the import works for now


def test_import_data_generation():
    """Test that data generation modules can be imported."""
    from ritini.data_generation import gene, sergio
    # Just test that the import works for now


def test_cli_import():
    """Test that CLI can be imported."""
    from ritini.cli import main
    # Just test that the import works for now


if __name__ == "__main__":
    test_import_ritini()
    test_import_core()
    test_import_data_generation()
    test_cli_import()
    print("All basic tests passed!")