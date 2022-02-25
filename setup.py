from distutils.core import setup

setup(
    name="metrics",
    version="0.1",
    description="A metrics package",
    author="AB",
    url="http://github.com/FLClab/metrics",
    packages=["metrics"],
    install_requires=[
        "scikit-learn",   
    ]
)
