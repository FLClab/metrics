import setuptools
from setuptools import setup

setup(
    name="metrics",
    version="0.1",
    description="A metrics package to quantify the performance of ML/DL methods",
    author="AB & GL",
    url="http://github.com/FLClab/metrics",
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "tifffile"
    ],
    package_dir={"": "."},
    include_package_data=True    
)
