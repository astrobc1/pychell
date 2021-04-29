import setuptools
import os

# Get requirements
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pychell",
    version="1.0.6",
    author="Bryson Cale",
    author_email="bryson.cale1@gmail.com",
    description="Toolkit from raw echelle spectra through orbit fitting.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages = setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    url="https://github.com/astrobc1/pychell",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix"
    ],
    python_requires='>=3.8'
)
