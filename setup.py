import setuptools

# version fetch
with open('kindling/version.py', 'r') as f:
    exec(f.read())

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='kinlin',
    version=__version__,
    description='WIP', # TODO
    long_description=long_description,
    long_description_content_type='text/markdown',
    # license='', # TODO
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
