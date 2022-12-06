import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("VERSION.txt", "r", encoding="utf-8") as f:
    version = f.read().strip()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("dev_requirements.txt") as f:
    dev_requirements = f.read().splitlines()

setuptools.setup(
    name="torchmps",
    version=version,
    author="Jacob Miller",
    author_email="jmjacobmiller@gmail.com",
    description="Pytorch library for matrix product state models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jemisjoky/TorchMPS",
    project_urls={
        "Bug Tracker": "https://github.com/jemisjoky/TorchMPS/issues",
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=requirements,
    extras_require={"development": set(dev_requirements)},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
