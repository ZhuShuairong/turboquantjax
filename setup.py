from setuptools import setup, find_packages

setup(
    name="turboquant-jax",
    version="0.1.0",
    description="JAX implementation of TurboQuant for ultra-fast KV cache quantization",
    author="TurboQuant Authors",
    packages=find_packages(include=["turboquant_jax", "turboquant_jax.*"]),
    py_modules=["turboquant", "compressors", "lloyd_max"],
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy",
        "scipy",
    ],
    extras_require={
        "test": [
            "pytest",
            "transformers",
            "torch",
            "bitsandbytes"
        ]
    },
    python_requires=">=3.8",
)
