"""Setup entry point for pip install."""
from setuptools import setup, find_packages

# All metadata is in pyproject.toml
setup(packages=find_packages(where="src"))
