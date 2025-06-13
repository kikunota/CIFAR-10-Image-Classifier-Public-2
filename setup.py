from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cv-image-classifier",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A PyTorch-based image classification system with web interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cv-image-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-classifier=src.training.train:main",
            "evaluate-classifier=src.training.evaluate:main",
            "classify-image=src.inference:main",
        ],
    },
)