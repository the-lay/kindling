import setuptools
import re

# version fetch
with open('kinlin/__init__.py', 'r') as f:
    version = re.search('__version__ = "(.+?)"', f.read()).group(1)

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name = 'kinlin',
    version = version,
    description = 'WIP: PyTorch based framework',  # TODO
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    author = 'the-lay',
    author_email = 'ilja@gubins.lv',
    url = 'https://github.com/the-lay/kinlin',
    keywords = [],  # TODO
    install_requires = [  # TODO
        'mrcfile', 'numpy'
    ],
    packages = setuptools.find_packages(),
    test_suite = 'tests',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ]
)
