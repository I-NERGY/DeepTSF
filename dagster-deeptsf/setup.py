from setuptools import find_packages, setup

setup(
    name="dagster_deeptsf",
    packages=find_packages(exclude=["dagster_deeptsf_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
