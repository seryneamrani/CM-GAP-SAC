"""Setup script for cm_gap_sac_navigation ROS 2 package.

Uses an explicit package list instead of find_packages() to avoid scanning
parasitic directories that may exist in the workspace (e.g. .egg-info dirs,
orphaned folders from previous builds).
"""
from setuptools import setup
from glob import glob
import os

package_name = 'cm_gap_sac_navigation'

# Explicit list: every Python sub-package we ship.
packages = [
    package_name,
    f'{package_name}.envs',
    f'{package_name}.models',
    f'{package_name}.perception',
    f'{package_name}.rl',
    f'{package_name}.training',
    f'{package_name}.utils',
]

setup(
    name=package_name,
    version='0.3.0',
    packages=packages,
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.24',
        'scipy>=1.10',
        'gymnasium>=0.29',
        'torch>=2.1',
        'osqp>=0.6.3',
    ],
    zip_safe=True,
    maintainer='Seryne Amrani',
    maintainer_email='seryne.amrani@estin.dz',
    description='CM-GAP_SAC: cross-modal gated attention SAC with CBF shield',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gym_smoke_test = cm_gap_sac_navigation.envs.limo_gazebo_env:smoke_test',
            'train = cm_gap_sac_navigation.training.train:main',
            'eval  = cm_gap_sac_navigation.training.eval:main',
        ],
    },
)
