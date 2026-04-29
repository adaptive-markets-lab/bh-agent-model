from setuptools import find_packages, setup


def parse_requirements(filename):
    """Load dependencies from a pip requirements file."""
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    name="bh-agent-model",
    version="0.0.0",
    description="All functionality used by bh agent model.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
)
