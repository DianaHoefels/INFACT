from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="infact",
    version="1.0.0",
    author="Diana Hoefels",
    description="A Romanian Institutional Fact-Checking Corpus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DianaHoefels/INFACT",
    packages=find_packages(where="src"),
    py_modules=["data_utils", "evaluate"],
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Romanian",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "infact-evaluate=evaluate:main",
        ],
    },
)
