from setuptools import setup, find_packages

setup(
    name="large-context-training",
    version="0.3.2",
    description="Framework for training large language models with extended context windows",
    author="Alex Santia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "flash-attn>=2.6.0",
        "triton>=3.0.0",
        "wandb>=0.17.0",
        "safetensors>=0.4.0",
        "accelerate>=0.30.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "datasets>=2.19.0",
        "einops>=0.8.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "black>=24.0", "ruff>=0.4.0"],
        "amd": ["rocrate>=0.8.0"],
    },
    entry_points={
        "console_scripts": [
            "lct-train=scripts.train:main",
            "lct-bench=scripts.benchmark_context:main",
        ],
    },
)