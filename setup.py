from setuptools import setup

setup(
    name='IRIXS',
    version='0.5.1',
    description='P01 IRIXS reduction routines',
    url='http://github.com/djlns/irixs',
    author='Joel Bertinshaw',
    author_email='djlns@posteo.net',
    license='GNU GPLv3',
    install_requirements=[
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
        'IRIXS/oneshot'
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
