import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="punppci",
    version="0.0.8",
    author="Jacky Poon",
    author_email="jackypn@gmail.com",
    description="Neural network for insurance claims modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JackyP/penalised-unexplainability-network-payments-per-claim-incurred",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
