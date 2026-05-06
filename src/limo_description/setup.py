from setuptools import setup

package_name = 'limo_description'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/display_limo.launch.py',
            'launch/limo_gazebo.launch.py',
            'launch/sim.launch.py',
            'launch/navigation.launch.py',
            'launch/bridge.launch.py',
        ]),
        ('share/' + package_name + '/urdf', ['urdf/limo.urdf.xacro']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seryne',
    maintainer_email='seryne@example.com',
    description='LIMO description and URDF',
    license='Apache License 2.0',
)
