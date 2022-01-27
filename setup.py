import os
import setuptools

with open(os.path.join(os.path.dirname(__file__),"predify/VERSION")) as f:
    version = f.read().strip() 


with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements

    
setuptools.setup(
    name="predify",
    version=version,
    author="miladmozafari",
    author_email="",
    description="Add Predictive Coding Dynamics to any Deep Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", # write github url,
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    python_requires='>=3.6',
)
