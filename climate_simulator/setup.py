from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize
ROOT = Path(__file__).parent.resolve()
CPP_SOURCES = [
    "src/animal_load.cpp",
    "src/config.cpp",
    "src/csv.cpp",
    "src/herd_inventory.cpp",
    "src/ifc_reader.cpp",
    "src/report.cpp",
    "src/simulator.cpp",
    "src/thermoregulation.cpp",
    "src/thermoregulation_config.cpp",
    "src/python_bridge.cpp",
]
ext_modules = cythonize([
    Extension(
        "climate_hall_simulator._core",
        sources=["climate_hall_simulator/_core.pyx", *CPP_SOURCES],
        include_dirs=[str(ROOT), str(ROOT / "include")],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
], compiler_directives={"language_level": "3"})
setup(
    name="climate_hall_simulator",
    version="0.2.0",
    packages=["climate_hall_simulator"],
    ext_modules=ext_modules,
    install_requires=["pandas>=2.0"],
    include_package_data=True,
)
