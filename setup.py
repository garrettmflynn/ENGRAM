import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="engram",
    version="0.0.1",
    author="Garrett Flynn",
    author_email="garrett@garrettflynn.com",
    description="Encoding Graphical Representations \
        of Activated Memories (ENGRAM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/garrettmflynn/Engram",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7',
)
