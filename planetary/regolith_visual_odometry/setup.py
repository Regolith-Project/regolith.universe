from setuptools import find_packages
from setuptools import setup

package_name = "regolith_visual_odometry"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Regolith Project contributors",
    maintainer_email="info@astro42.com",
    description="RGB-D visual odometry publishing body-frame velocity, so the estimator can observe lateral slip",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "visual_odometry_node = regolith_visual_odometry.visual_odometry_node:main",
        ],
    },
)
