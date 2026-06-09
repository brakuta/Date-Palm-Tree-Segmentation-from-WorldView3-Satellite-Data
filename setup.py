from setuptools import setup, find_packages

setup(
    name='date-palm-tree-segmentation-from-worldview3-satellite-data',
    version='0.1.0',
    description='Date palm semantic segmentation & individual-tree mapping '
                'from WorldView-3 (and RGB) imagery.',
    packages=find_packages(include=['palmseg', 'palmseg.*']),
    python_requires='>=3.9',
    entry_points={'console_scripts': ['palmseg = palmseg.cli:main']},
    license='Apache-2.0',
)
