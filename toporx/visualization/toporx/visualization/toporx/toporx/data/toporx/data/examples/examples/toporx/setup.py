"""
TopoRx: Topological Biomarker Discovery
========================================

A Python package for predicting cancer drug response
using Topological Data Analysis (TDA).

Installation:
    pip install -e .

Author: Angelica Alvarez
GitHub: https://github.com/aalvarez122/TopoRx
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="toporx",
    version="0.1.0",
    author="Angelica Alvarez",
    author_email="neurosalvarez@arizona.edu",
    description="Predicting cancer drug response using Topological Data Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aalvarez122/TopoRx",
    project_urls={
        "Bug Tracker": "https://github.com/aalvarez122/TopoRx/issues",
        "Documentation": "https://github.com/aalvarez122/TopoRx#readme",
        "Source Code": "https://github.com/aalvarez122/TopoRx",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "topological-data-analysis",
        "TDA",
        "persistent-homology",
        "drug-response",
        "cancer",
        "bioinformatics",
        "machine-learning",
        "biomarker-discovery",
        "precision-medicine"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "full": [
            "gudhi>=3.8.0",
            "persim>=0.3.1",
            "ripser>=0.6.4",
            "plotly>=5.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "toporx-demo=toporx.demo:run",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
