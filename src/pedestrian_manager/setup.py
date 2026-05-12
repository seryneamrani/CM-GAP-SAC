from setuptools import setup
from glob import glob
import os

package_name = "pedestrian_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Seryne Amrani",
    maintainer_email="seryne@estin.dz",
    description="Gestion Python des pietons dynamiques",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pedestrian_manager_node = pedestrian_manager.pedestrian_manager_node:main",
        ],
    },
)