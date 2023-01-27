import setuptools

setuptools.setup(
    name="anaconda.mlflow.tracking.sdk",
    version="0.10.1",
    package_dir={"": "src"},
    packages=setuptools.find_namespace_packages(where="src"),
    author="Joshua C. Burt",
    description="Anaconda MLFlow Tracking SDK",
    long_description="Anaconda MLFlow Tracking SDK",
    include_package_data=True,
    zip_safe=False,
)
