from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='opentensor',
    version='0.0.2',
    description='Opentensor',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/opentensor/opentensor',
    author='opentensor.org',
    author_email='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(
        exclude=['data', 'contract', 'assets', 'scripts', 'docs']),
    python_requires='>=3.5',
    install_requires=[
        'grpcio', 'google-api-python-client', 'numpy', 'pickle-mixin'
    ],  # Optional
)
