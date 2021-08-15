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
        "webdataset",
    ],
)
