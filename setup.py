import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='wheat-filtration-faunam',
    version='0.1',
    author='faunam & xandaschofield',
    author_email='xanda@cs.hmc.edu',
    url='www.github.com/faunam/wheat_filtration',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['wheat_filtration'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    )
