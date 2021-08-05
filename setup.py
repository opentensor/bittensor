from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from os import path
from io import open
import codecs
import re
import os

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
with open('requirements.txt') as requirements_file:
    install_requires = [str(requirement) for requirement in parse_requirements(requirements_file)]

# loading version from setup.py
with codecs.open(os.path.join(here, 'bittensor/__init__.py'), encoding='utf-8') as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

setup(
    name='bittensor',
    version=version_string,
    description='bittensor',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/opentensor/bittensor',
    author='bittensor.com',
    packages=find_packages(),
    include_package_data=True,
    author_email='',
    license='MIT',
    install_requires=install_requires,
    scripts=['bin/bittensor', 'bin/bittensor-cli'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.5',
)
