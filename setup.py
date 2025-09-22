"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="unimol_tools",
    version="0.1.5",
    description=(
        "unimol_tools is a Python package for property prediction with Uni-Mol in molecule, materials and protein."
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="DP Technology",
    author_email="unimol@dp.tech",
    license="The MIT License",
    url="https://github.com/deepmodeling/unimol_tools",
    packages=find_packages(
        where='.',
        exclude=[
            "build",
            "dist",
        ],
    ),
    install_requires=[
        "numpy<2.3.0,>=2.0.0",
        "pandas>=2.2.2",
        "torch>=2.4.0",
        "joblib",
        "rdkit>=2024.3.4",
        "pyyaml",
        "addict",
        "scikit-learn>=1.5.0",
        "numba",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "tensorboard",
        "lmdb",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
