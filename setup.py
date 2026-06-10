from setuptools import setup, find_packages

# Core runtime dependencies that pip can resolve directly. The MM stack
# (mmcv/mmengine/mmsegmentation/mmdet) and torch are intentionally NOT listed:
# they must be installed via MIM against the user's specific CUDA build (see
# docs/INSTALL.md). Extraction-only use (mode b) needs only the packages below.
INSTALL_REQUIRES = [
    'numpy>=1.24',
    'rasterio>=1.3',
    'geopandas>=0.13',
    'shapely>=2.0',
    'scipy>=1.10',
    'scikit-image>=0.21',
    'opencv-python-headless>=4.7',
    'Pillow>=9.5',
    'tqdm>=4.65',
    'huggingface_hub>=0.20',
    'safetensors>=0.4',
]

setup(
    name='date-palm-tree-segmentation-from-worldview3-satellite-data',
    version='0.1.0',
    description='Date palm semantic segmentation & individual-tree mapping '
                'from WorldView-3 (and RGB) imagery.',
    packages=find_packages(include=['palmseg', 'palmseg.*']),
    python_requires='>=3.9',
    install_requires=INSTALL_REQUIRES,
    entry_points={'console_scripts': ['palmseg = palmseg.cli:main']},
    license='Apache-2.0',
)
