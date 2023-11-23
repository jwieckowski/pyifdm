import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyifdm",
    version="1.1.0",
    author="Jakub Więckowski",
    author_email="J.Wieckowski@il-pib.pl",
    description="Python library to support Decision Making with Intuitionistic Fuzzy Sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwieckowski/pyifdm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy'
    ]
)
