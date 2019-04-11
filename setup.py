import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="punppci",
    version="0.0.1",
    author="Jacky Poon",
    author_email="jackypn@gmail.com",
    description="Neural network for insurance claims modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/jpoon/punppci",
    packages=setuptools.find_packages(),
    install_requires=["keras"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
