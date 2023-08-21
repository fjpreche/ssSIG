from setuptools import setup, find_packages

from ssSIG import __version__

setup(
    name='ssSIG',
    version=__version__,

    url='https://github.com/MichaelKim0407/tutorial-pip-package',
    author='Francisco Perez-Reche',
    author_email='fperez-reche@abdn.ac.uk',

    packages=find_packages(),
)
