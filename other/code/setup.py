from setuptools import setup, find_packages

setup(
  name='main',
  version='0.8.0',
  author='Artin Majdi',
  author_email='mohammadsmajdi@arizona.edu',
  description='PhD code',
  packages=find_packages(include=['main' , 'main.*']),
  # install_requires = ['numpy', 'pandas', 'scipy'],
)


# command: pip install -e .
