import codecs
import os
import pathlib
import re
from io import open
from os import path

from setuptools import setup, find_packages


def read_requirements(path_):
    requirements_ = []

    with pathlib.Path(path_).open() as requirements_txt:
        for line in requirements_txt:
            if line.startswith("git+"):
                pkg_name = re.search(r"egg=([a-zA-Z0-9_-]+)", line.strip()).group(1)
                requirements_.append(pkg_name + " @ " + line.strip())
            else:
                requirements_.append(line.strip())

    return requirements_


requirements = read_requirements("requirements/prod.txt")
extra_requirements_dev = read_requirements("requirements/dev.txt")
extra_requirements_cubit = read_requirements("requirements/cubit.txt")
extra_requirements_torch = read_requirements("requirements/torch.txt")
extra_requirements_cli = read_requirements("requirements/cli.txt")

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# loading version from setup.py
with codecs.open(
    os.path.join(here, "bittensor/core/settings.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="bittensor",
    version=version_string,
    description="bittensor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opentensor/bittensor",
    author="bittensor.com",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={
        "bittensor": ["utils/certifi.sh"],
    },
    author_email="",
    license="MIT",
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": extra_requirements_dev + extra_requirements_torch,
        "torch": extra_requirements_torch,
        "cli": extra_requirements_cli,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
