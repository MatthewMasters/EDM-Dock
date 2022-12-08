from setuptools import setup, find_packages

setup(
    name='EDM-Dock',
    version='1.0',
    description='EDM-Dock',
    author='Matthew Masters',
    author_email='matthew.masters@unibas.ch',
    packages=find_packages(include=['edmdock', 'edmdock.*']),
    include_package_data=True,
    python_requires='>=3.9.5'
)
