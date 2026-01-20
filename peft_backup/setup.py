from setuptools import setup, find_packages

setup(
    name="peft",
    version="0.18.0+lorahub",
    description="PEFT: State-of-the-art Parameter-Efficient Fine-Tuning (Modified for LoraRetriever)",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers",
        "accelerate>=0.21.0",
        "safetensors",
        "huggingface-hub>=0.17.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
