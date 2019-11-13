from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "pandas", "scipy"]

setup(
    name="croston",
    version="0.0.1",
    author="abc",
    author_email="abc@gmail.com",
    description="croston model for intermittent time series",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/croston/homepage/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)