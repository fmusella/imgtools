from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="imgtools",
    version="0.43+fix_viz",
    author="Francesco Musella",
    author_email="fmusella@g.ucla.edu",
    description="A set of tools for single-cell DNA imaging processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    python_requires='>=3.7.3',
    install_requires=[
        "numpy>=1.20.3",
        "alabtools>=1.1.13",
        "pydantic>=2.4.2",
        "trimesh>=3.21.5",
        "alphashape>=1.3.1",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.3",
        "scipy>=1.7.3",
        "h5py>=3.7.0",
        "statsmodels>=0.13.5",
        "mrcfile>=1.5.0",
    ],
    entry_points={
        # If you have any scripts or command line tools you can add them here
        # 'console_scripts': [
        #     'myscript=imgtools.myscript:main',
        # ],
    }
)
