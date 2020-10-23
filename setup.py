#!/usr/bin/env python3

import sys

from setuptools import setup, find_packages

VERSION = '0.0.1'

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >=3.6 is required.')

with open('README.md', encoding="utf8") as f:
    # strip the header and badges etc
    readme = f.read().split('--------------------')[-1]

# This is currently blank as currently running from a conda env.
with open('requirements.txt') as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split('==')[0])


if __name__ == '__main__':
    setup(
        name='story-fragments',
        version=VERSION,
        description='Models to experiment with story understanding and'
                    ' generation tasks using embedding based knowledgebase and memory representations.',
        long_description=readme,
        long_description_content_type='text/markdown',
        python_requires='>=3.6',
        packages=find_packages(
            exclude=('data', 'docs', 'examples', 'tests')
        ),
        install_requires=reqs,
        include_package_data=True,
        package_data={'': ['*.txt', '*.md']},
        entry_points={
            "console_scripts": ["story_fragments=story_fragments.__main__:main"],
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: NLP",
            "Natural Language :: English",
        ],
    )
