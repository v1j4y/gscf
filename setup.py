#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="vijay",
    author_email='vijay.gopal.c@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="GSCF",
    entry_points={
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='gscf',
    name='gscf',
    packages=find_packages(include=['gscf', 'gscf.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/v1j4y/gscf',
    version='0.1.0',
    zip_safe=False,
)
