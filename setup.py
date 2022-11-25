# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

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
    scripts=['bin/btcli'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    extras_requires={
        'cubit': ['cubit>=1.1.0 @ git+https://github.com/opentensor/cubit.git']
    }
)
