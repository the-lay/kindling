import setuptools
import re

# version fetch
with open('kinlin/__init__.py', 'r') as f:
    version = re.search('__version__ = "(.+?)"', f.read()).group(1)

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='kinlin',
    version=version,
    description='WIP', # TODO
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='BSD',
    author='the-lay',
    author_email='ilja.gubin@gmail.com',
    url='https://github.com/the-lay/kinlin',
    platforms=['any'],
    install_requires=[ # TODO
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    test_suite='tests'
)
