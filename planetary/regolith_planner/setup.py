from setuptools import find_packages, setup

package_name = "regolith_planner"

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
    description="Cost-aware A* global planner over the traversability costmap, with path smoothing",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planner_node = regolith_planner.planner_node:main",
        ],
    },
)
