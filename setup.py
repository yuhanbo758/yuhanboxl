from setuptools import setup, find_packages

setup(
    name='yuhanboxl',
    version='0.2.0',
    packages=find_packages(),
    description='效率工具，调用ai的api等',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yuhanbo758/yuhanboxl',
    author='余汉波',
    author_email='yuhanbo@sanrenjz.com',
    license='MIT',
    install_requires=[
        'requests',
        'numpy', 
        'pandas', 
        'openai',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # 根据您的开发状态选择：Alpha/Beta/Stable
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 根据需要修改
)
