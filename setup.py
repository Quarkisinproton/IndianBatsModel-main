from setuptools import setup, find_packages

setup(
    name='bat-species-classifier',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A deep learning model to classify bat species using voice recordings converted to spectrogram images.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'librosa',
        'scikit-learn',
        'PyYAML',
        'Jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)