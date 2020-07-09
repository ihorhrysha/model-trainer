import setuptools


requirements = [
    line for line in (line.strip() for line in open('requirements.txt'))
    if line and not (line.startswith("#"))
    ]


setuptools.setup(
    name='model-trainer',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
)
