import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
	
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pyVPLM",
    version="0.2",
	install_requires=requirements,
    author="Aurelien Reysset",
    author_email="aurelien.reysset@insa-toulouse.fr",
    description="Variable Power-Law regression Models tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SizingLab/methods_and_tools/pyvplm-master",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
