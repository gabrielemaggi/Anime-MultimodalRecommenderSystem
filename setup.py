from setuptools import setup, find_packages

setup(
    name="anime_recommender_project",
    version="0.1.0",
    author="Omer , Nicolo, Allesandro e Gabriele",
    description="A multimodal recommendation system for Anime",

    packages=find_packages(),

    python_requires=">=3.8",

    install_requires=[
        "numpy",
        "sentence-transformers",
        "torch",
        "torchvision",
        "transformers",
    ],
)