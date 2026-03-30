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
        "psutil>=5.9.0",
    ],
    extras_require={
        "hf": [
            "transformers>=4.40.0",
            "torch>=2.2.0",
            "accelerate>=0.25.0",
            "sentencepiece>=0.1.99",
        ],
        "gguf": [
            "llama-cpp-python>=0.2.78",
        ],
        "test": [
            "pytest",
            "transformers",
            "torch",
            "bitsandbytes",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "tqjax-generate=turboquant_jax.runtime.entrypoints:tqjax_generate_main",
            "tqjax-bench=turboquant_jax.runtime.entrypoints:tqjax_bench_main",
            "tqjax-serve=turboquant_jax.runtime.entrypoints:tqjax_serve_main",
            "tqjax-validate=turboquant_jax.runtime.entrypoints:tqjax_validate_main",
            "tqjax-env=turboquant_jax.runtime.entrypoints:tqjax_env_main",
        ]
    },
    python_requires=">=3.8",
)
