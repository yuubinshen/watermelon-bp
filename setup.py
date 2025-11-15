from setuptools import setup, find_packages

setup(
    name="watermelon-bp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="标准BP神经网络分类器（适配西瓜数据集3.0）",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/woqunimad/watermelon-bp", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)