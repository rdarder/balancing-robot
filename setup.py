from setuptools import setup, find_packages

setup(
    name="balancing_robot",  # Replace with your package name
    version="0.1.0",  # Replace with your version
    packages=find_packages(),  # This automatically finds all packages
    # Other metadata (author, license, etc.) can go here
    include_package_data=True,
        package_data={
            "balancing_robot": ["assets/**/*", "checkpoints/**/*"]
        }
)
