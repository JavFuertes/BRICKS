from setuptools import setup, find_packages

setup(
    name="bricks",
    version="0.1.0",
    packages=find_packages(include=['bricks*'], exclude=[
        'bricks._example*',
        '.github*'
    ]),
    install_requires=[
        'numpy',
        'pandas', 
        'scipy',
        'tabulate',
        'matplotlib',
        'plotly',
        'dash',
        'scikit-learn',
        'scienceplots',
    ],           
    python_requires='>=3.8',  
    description='Tools for the assessment of masonry structures'          
) 