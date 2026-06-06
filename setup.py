from setuptools import find_packages, setup

setup(
    name="lake-water-level-forecast",
    version="0.1.0",
    description="NWCL1 wind-driven water level forecast workflow",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
