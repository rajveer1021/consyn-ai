import os
from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Filter out comments and platform-specific requirements
requirements = [r for r in requirements if not r.startswith("#") and ";" not in r]

# Read long description from README
if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()
else:
    long_description = "Consyn AI - A family of language models"

setup(
    name="consyn",
    version="0.1.0",
    description="Consyn AI - A family of language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Consyn AI Team",
    author_email="support@consyn.ai",
    url="https://github.com/consynai/consyn",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "consyn-train=consyn.scripts.train:main",
            "consyn-evaluate=consyn.scripts.evaluate:main",
            "consyn-generate=consyn.scripts.generate:main",
            "consyn-convert=consyn.scripts.convert_checkpoint:main",
            "consyn-serve=consyn.api.main:serve",
        ],
    },
)
