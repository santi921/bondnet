from setuptools import setup, find_packages
from pathlib import Path


def get_version(fname=Path.cwd().joinpath("bondnet", "__init__.py")):
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if "__version__" in line:
                v = line.split("=")[1]
                if "'" in v:
                    version = v.strip("' ")
                elif '"' in v:
                    version = v.strip('" ')
                break
    return version


setup(
    name="bondnet",
    version=get_version(),
    packages=find_packages(),
    entry_points={"console_scripts": ["bondnet = bondnet.scripts.predict_cli:cli"]},
    install_requires=["numpy", "pyyaml", "scikit-learn", "click"],
    author="Mingjian Wen, Santiago Vargas",
    author_email="wenxx151@gmail.com, santiagovargas921@gmail.com",
    url="https://github.com/santi921/bondnet",
    description="short description",
    long_description="long description",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "./bondnet/scripts/training/lightning_bayes_opt.py",
        "./bondnet/scripts/training/lightning_train.py",
        "./bondnet/scripts/training/lightning_controller.py",
        "./bondnet/scripts/training/lightning_generate_settings.py",
        "./bondnet/scripts/training/launch_sbatch.py",
    ],
    zip_safe=False,
)
