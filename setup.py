from setuptools import setup


setup(
    name='titanic',
    version='0.1.0',
    packages=['titanic'],
    package_dir={'titanic': 'src'},
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'seaborn'
    ],
)
