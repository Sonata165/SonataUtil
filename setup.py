from setuptools import setup, find_packages
import os


version = '0.0.1'

readme_path = 'Readme.md'
long_description = open(readme_path, encoding='utf-8').read() if os.path.exists(readme_path) else ""


setup(
    name='SonataUtil',  # 项目名称
    version=version,  # 版本号
    author='Longshen Ou',  # 作者姓名
    author_email='oulongshen@gmail.com',  # 邮箱地址
    description="Longshen's utility functions",  # 项目简介
    long_description=open('Readme.md').read(),  # 从 Readme.md 加载详细描述
    long_description_content_type='text/markdown',  # README 格式
    url='https://github.com/Sonata165/SonataUtil',  # 项目主页 URL
    packages=find_packages(),  # 自动查找所有包含 `__init__.py` 的包
    install_requires=[  # 项目的依赖项
        'pyyaml>=6.0.2',
    ],
    classifiers=[  # 分类器，描述项目的适用性
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
    ],
    python_requires='>=3.7',  # 支持的最低 Python 版本
    license='MIT',  # 项目许可证
)