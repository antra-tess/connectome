from setuptools import setup, find_packages

setup(
    name="bot_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-socketio",
        "python-dotenv",
        "pytest",
        "pytest-mock",
    ],
    python_requires=">=3.8",
) 