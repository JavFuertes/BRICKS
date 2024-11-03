from setuptools import setup, find_packages

setup(
    name="bricks",
    version="0.1.0",
    packages=find_packages(exclude=['_example*']),
    install_requires=[
        'numpy',
        'pandas', 
        'scipy',
        'tabulate',
        'matplotlib',
        'torch',
        'botorch',
        'plotly',
        'dash',
        'scikit-learn',
    ],           
    python_requires='>=3.7',  
    description='Package for brick wall analysis and calibration'          
)