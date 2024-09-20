from setuptools import setup, find_packages

setup(name="inferx",
      version="1.0.1",
      author="yuanyang",
      author_email="yuan823978@gmail.com",
      url="https://github.com/yuanyang1991/InferX",
      packages=find_packages(),
      description="deep learning deploy framework",
      python_requires='>=3.6',
      install_requires=[
          'tensorrt==10.3.0',
      ]
      )
