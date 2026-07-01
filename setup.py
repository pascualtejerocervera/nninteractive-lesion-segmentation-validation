import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

setup(
    install_requires=[
        f"nnInteractive_v1 @ file://{here}/models/nninteractive_v1",
        f"nnInteractive_v2 @ file://{here}/models/nninteractive_v2",
        "surface-distance @ git+https://github.com/google-deepmind/surface-distance.git",
        "numpy<2.0",
        "scipy>=1.10",
        "scikit-image>=0.21",
        "nibabel>=5.0",
    ],
)
