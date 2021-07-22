from setuptools import setup

setup(
    name="hippocampy",
    version="0.1",
    description="Toolbox for data anaylsis in python",
    author="Bourboulou Romain",
    author_email="bouromain@gmail.com",
    packages=["hippocampy"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "astropy",
        "scikit-image",
        "bottleneck",
        "suite2p",
        "scanimage-tiff-reader",
        "h5py",
        "mat73",
        "sklearn",
        "tqdm",
        "pytest",
        "numba",
        "cython",
        "oasis @ git+ssh://git@github.com/j-friedrich/OASIS",
    ],
)
