# Setup Python executable
if(NOT DEFINED Python3_FIND_REGISTRY)
    # Only consider PATH variable on Windows by default
    set(Python3_FIND_REGISTRY NEVER)
endif()
# Requires Python 3.6+
find_package(Python3 3.6 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    # Setup PYTHON_EXECUTABLE for 3rdparty modules which still use the deprecated find_package(PythonInterp)
    set(PYTHON_EXECUTABLE
        ${Python3_EXECUTABLE}
        CACHE STRING "Deprecated path to the Python executable (for 3rdparty only)" FORCE)
else()
    message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python >= 3.6. Please ensure it is in PATH.")
endif()

if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python 3 not found in top level file")
endif()

# We need to get python version to configure some meta files
set(PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
set(PYPI_PACKAGE_NAME "vega")


pybind11_add_module(vegapy)

set_target_properties(
    vegapy
    PROPERTIES FOLDER "vegapy"
               LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

add_definitions(-DNEW_MODEL=1 -DUSE_UUID=1)
add_dependencies(vegapy build-vega ext_ffmpeg ext_gb28181 ext_mspout)

list(APPEND SRC_FILES vegapy.cpp vegapy_bind.cpp video.cpp video_bind.cpp)
target_sources(vegapy PRIVATE ${SRC_FILES})

list(APPEND INCDIRS ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "includes: ${INCDIRS}")

include_directories(${INCDIRS})
target_link_libraries(vegapy PRIVATE ${DEPLIBS} glog)

set(PYTHON_PACKAGE_SRC_DIR "${PROJECT_SOURCE_DIR}/python")
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")

set(BUILD_DG_BASE_INFER $ENV{DG_BASE_INFER})
set(VEGA_BUILD_TYPE ${CMAKE_BUILD_TYPE})
message(STATUS "VEGA_BUILD_TYPE is ${VEGA_BUILD_TYPE}")
configure_file("_build_config.py.in" "${CMAKE_BINARY_DIR}/lib/_build_config.py.in")
file(
    GENERATE
    OUTPUT "${PYTHON_COMPILED_MODULE_DIR}/_build_config.py"
    INPUT "${CMAKE_BINARY_DIR}/lib/_build_config.py.in")

set(GENERATED_OUTPUTS "")
set(COMPILED_MODULE_PATH_LIST $<TARGET_FILE:vegapy>)

add_custom_target(
    vegas_proto
    COMMAND echo "compile vegas proto to python"
    COMMAND echo protoc -I=${VEGA_PROTO_DIR} --python_out=${PYTHON_COMPILED_MODULE_DIR} ${VEGA_PROTO_DIR}/vegas.proto
    COMMAND protoc -I=${VEGA_PROTO_DIR} --python_out=${PYTHON_COMPILED_MODULE_DIR} ${VEGA_PROTO_DIR}/vegas.proto
    DEPENDS ${VEGA_PROTO_DIR}/vegas.proto build-vega)

add_custom_target(
    python-package
    COMMAND
        ${CMAKE_COMMAND} -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR} -DPYTHON_COMPILED_MODULE_DIR=${PYTHON_COMPILED_MODULE_DIR}
        -DPROJECT_ROOT_DIR=${CMAKE_SOURCE_DIR} -DPROJECT_EMAIL=${PROJECT_EMAIL} -DPYTHON_VERSION=${PYTHON_VERSION}
        "-DCOMPILED_MODULE_PATH_LIST=${COMPILED_MODULE_PATH_LIST}" "-DPYTHON_EXTRA_DIRS=${PYTHON_EXTRA_DIRS}"
        -DPROJECT_HOMEPAGE_URL=${PROJECT_HOMEPAGE_URL} -DBUILD_VEGA_VERSION=${VEGA_VERSION} -DVEGA_TARGET=${VEGA_TARGET}
        -DVEGA_BASE_INFER=${VEGA_BASE_INFER} -DVEGA_ARCH=${VEGA_ARCH}
        # -DPROJECT_DOCS=${PROJECT_DOCS} -DPROJECT_CODE=${PROJECT_CODE}
        -DPROJECT_VERSION=${VEGAPY_VERSION} -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
    VERBATIM
    DEPENDS ${GENERATED_OUTPUTS} vegas_proto)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(
    pip-package
    COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package)

# Use `make install-pip-package` to install pip wheel package to the current python environment.
add_custom_target(
    install-pip-package
    COMMAND ${CMAKE_COMMAND} -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE} -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package)

# FOR DEBUGGING ONLY Use `make install-python-package` to build and install python package in the current python
# environment. This is substantially faster than `make install-pip-package`. However this approach does not create wheel
# or egg files and does not take care of dependencies thus not suitable for deployment. Ref:
# https://stackoverflow.com/a/33791008/1255535
add_custom_target(
    install-python-package
    COMMAND ${Python3_EXECUTABLE} setup.py install --single-version-externally-managed --root=/
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package)
