from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sbanetwork',
    version='0.1.2', 
    description='Lightweight Neural Network Library from scratch',
    long_description=long_description,
    long_description_content_type='text/markdown',  
    author='Abhishek Gupta',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
