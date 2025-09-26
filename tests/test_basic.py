"""
Basic tests for RiTINI package functionality.
"""

import pytest


def test_import_RiTINI():
    """Test that the package can be imported."""
    import RiTINI
    assert RiTINI.__version__ == "0.1.0"


def test_import_core():
    """Test that core modules can be imported."""
    from RiTINI.ritini import RiTINI
    # Just test that the import works for now


def test_import_data_generation():
    """Test that data generation modules can be imported."""
    from RiTINI.data_generation import gene, sergio
    # Just test that the import works for now


if __name__ == "__main__":
    test_import_RiTINI()
    test_import_core()
    test_import_data_generation()
    print("All basic tests passed!")