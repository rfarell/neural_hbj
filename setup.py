from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-hjb",
    version="1.1.0",
    author="Ryan Farell",
    description="Neural network solutions for Hamilton-Jacobi-Bellman equations in optimal control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rfarell/neural_hbj",
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={"dev": ["pytest"]},
)
