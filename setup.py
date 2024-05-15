from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
PACKAGE_ROOT = ROOT / "olaf"

with open("requirements.txt", "r") as f:
    requirements = [pac[:-1] for pac in f.readlines()]

setup(
    name="olaf",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
      entry_points={
        'console_scripts': [
            'olaf=olaf.__main__:main',
        ]
    }
)
