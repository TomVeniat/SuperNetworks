import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='supernets',
      version='0.0.6',
      description='A module containing the basics to use Super Networks',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Tom Veniat',
      packages=setuptools.find_packages(),
      license="MIT License",
      url='https://github.com/TomVeniat/SuperNetworks')
