# setup.py
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    description = fh.read()

setup(
    name="herbSyner_Finder",
    version="0.1.0",
    author="Yinyin Wang",
    author_email="yinyin.wang@cpu.edu.cn",
    packages=["herbSyner_Finder"],
    description="Synergistic Ingredient Discovery in Herbal Medicine",
    long_description=description,
    url="https://github.com/19900321/HerbSyner_Finder",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "pandas",
        "numpy",
        "networkx",
        "communities"
    ],
)