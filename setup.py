from setuptools import setup

setup(
    name="SparseSuit",
    version="0.1.0",
    author="Nicola Küng",
    packages=[
        "sparsesuit",
        "sparsesuit.data_generation",
        "sparsesuit.learning",
        "sparsesuit.utils",
        "sparsesuit.constants",
    ],
    install_requires=[
        "numpy",
        "torch",
        "tensorboard>=1.14.0",
        "hydra-core",
        "smplx",
        "pyrender",
        "trimesh",
        "welford",
        "opencv-python",
        "numpy-quaternion",
        "numba",
        "hydra-submitit-launcher",
    ],
)
