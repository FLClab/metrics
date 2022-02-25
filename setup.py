import setuptools
from setuptools import setup

print(setuptools.find_packages(where="metrics"))

setup(
    name="metrics",
    version="0.1",
    description="A metrics package",
    author="AB",
    url="http://github.com/FLClab/metrics",
#     packages=["metrics"],
    packages=setuptools.find_packages(where="metrics"),
    install_requires=[
        "scikit-learn",   
    ]
)
