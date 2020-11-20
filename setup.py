import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="predify-miladmozafari", # Replace with your own username
    version="0.0.1",
    author="miladmozafari",
    author_email="",
    description="Add Predictive Coding Dynamics to any Deep Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", # write github url,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)