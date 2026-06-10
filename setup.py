from setuptools import setup, find_packages

setup(
    name="spacecraft-fm",
    version="0.1.0",
    description="Foundation Model for Spacecraft Attitude Dynamics and Disturbance Torque Estimation",
    author="Fawzan1010",
    author_email="your-email@example.com",
    url="https://github.com/Fawzan1010/SpacecraftFM",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "mlflow>=2.8.0",
        "earthaccess>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch-cuda>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spacecraft-fm=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
