from setuptools import find_packages, setup

package_name = "regolith_terrain_gen"

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
    description="Procedural lunar terrain generator: heightmaps, crater fields, rock scatter, and Gazebo world files",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "generate_terrain = regolith_terrain_gen.cli:main",
        ],
    },
)
