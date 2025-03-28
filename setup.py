# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='galpy',
    version="0.1",
    url='https://github.com/ckarwin/galpy',
    author='Chris Karwin',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    install_requires = ['astropy','numpy','healpy','mhealpy','matplotlib','pandas'],
    description = 'all-sky map handling for MeV astronomy',
    entry_points = {"console_scripts":["new_galpy = galpy.make_new:main"]}
)
