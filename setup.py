from setuptools import setup

long_description = """A library that provides a wrapper for an R implementation of the random forest and bagging ensemble algorithms utilizing conditional
    inference trees as base learners. Includes a permutation importance helper function.
"""

setup(
    name='pycforest',
    version='0.1',
    url='https://github.com/gavin-s-smith/pycforest',
    license='MIT',
    py_modules=['pycforest'],
    author='Gavin Smith',
    author_email='gavin.smith@nottingham.ac.uk',
    install_requires=['numpy','pandas','sklearn','rpy2'],
    description='Wrapper for the Conditional Random Forests in R including permutation importance methods.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='scikit-learn random forest feature permutation importances',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
