from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['docopt']

setup(
  name='my-package',
  version='0.1',
  author = 'Sean Zhang',
  author_email = 'seanzhang2001@gmail.com',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  description='An example package for training on Cloud ML Engine.')