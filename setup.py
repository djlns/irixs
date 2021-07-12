from setuptools import setup

with open('README.md') as readme_file:
    README = readme_file.read()

setup(
    name='IRIXS',
    version='0.6.5',
    description='Routines for the IRIXS spectrometer',
    long_description=README,
    long_description_content_type="text/markdown",
    url='http://github.com/djlns/irixs',
    author='Joel Bertinshaw',
    author_email='djlns@posteo.net',
    license='GNU GPLv3',
    install_requires=[
        'numpy>=1.18',
        'matplotlib>=3.1',
        'scipy>=1.4',
        'scikit-image>=0.16',
        'PyQt5>=5.9'
    ],
    zip_safe=False,
    packages=['IRIXS'],
    scripts=[
        'IRIXS/p01plot',
        'IRIXS/irixs_oneshot'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Visualization'
    ]
)
