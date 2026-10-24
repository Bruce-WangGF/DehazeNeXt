from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
REQUIREMENTS = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]

setup(
    name='dehazenext',
    version='0.1.0',
    description='DehazeNeXt image dehazing network and BasicSR training pipeline',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires='>=3.8',
)
