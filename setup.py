import os
from runpy import run_path

from setuptools import find_packages, setup

# read the program version from version.py (without loading the module)
__version__ = run_path("src/omunet/version.py")["__version__"]


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="UTF-8").read()


setup(
    name="omunet",
    version=__version__,
    author="Jerome Wuerf",
    author_email="jw20qave@uni-leipzig.de",
    description="A short summary of the project",
    license="proprietary",
    url="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"omunet": ["res/*"]},
    long_description=read("README.md"),
    install_requires=[],
    tests_require=[
        "pytest",
        "pytest-cov",
        "pre-commit",
    ],
    platforms="any",
    python_requires=">=3.7",
)
