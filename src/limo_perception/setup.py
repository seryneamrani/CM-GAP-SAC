from setuptools import setup

package_name = 'limo_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '.models'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seryne',
    maintainer_email='seryne@example.com',
    description='LIMO perception : YOLOv8 + DeepSORT, projection 3D, prediction de trajectoire',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'perception_node = limo_perception.perception_node:main',
            'obstacle_projector = limo_perception.obstacle_projector:main',
            'scene_describer = limo_perception.scene_describer:main',
            'trajectory_predictor_node = limo_perception.trajectory_predictor_node:main',
        ],
    },
)
