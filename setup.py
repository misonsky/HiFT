import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HiFT",
    version="0.0.2",
    author="Yongkang Liu",
    author_email="misonsky@163.com",
    description="PyTorch implementation of 'HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy', a memory-efficient approach to adapt a large pre-trained deep learning model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/misonsky/HiFT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
