from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'limo_evaluation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Seryne',
    maintainer_email='seryne@example.com',
    description='Evaluation harness for LIMO Pro',
    license='MIT',
    entry_points={
        'console_scripts': [
            'evaluation_node = limo_evaluation.evaluation_node:main',
            'goal_runner = limo_evaluation.goal_runner:main',
            'pose_picker = limo_evaluation.pose_picker:main',
        ],
    },
)
