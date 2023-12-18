cmake_minimum_required(VERSION "3.2" FATAL_ERROR)
project(dgtrt)

set(DGTRT_VERSION "0.1")
set(PROJECT_EMAIL "guoqingjiang@deepglint.com")
include(cmake/compile.cmake)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<$<CONFIG:Debug>:Release>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<$<CONFIG:Debug>:Release>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# At `make`:  (or the equivalents) will be created at PYTHON_COMPILED_MODULE_DIR. The default location is
# `build/lib/${CMAKE_BUILD_TYPE}/`
set(PYTHON_COMPILED_MODULE_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/compiled")
message(STATUS "PYTHON_COMPILED_MODULE_DIR is ${PYTHON_COMPILED_MODULE_DIR}")
set(PYTHON_EXTRA_DIRS "")

find_package(pybind11 REQUIRED)

if(NOT WIN32 AND NOT DEFINED USE_CXX11_ABI)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE USE_CXX11_ABI)
  # Convert the bool variable to integer.
  if(USE_CXX11_ABI)
    set(USE_CXX11_ABI 1)
  else()
    set(USE_CXX11_ABI 0)
  endif()
  message(STATUS "USE_CXX11_ABI is set by python Torch to ${USE_CXX11_ABI}")
endif()
if(NOT USE_CXX11_ABI)
  add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
endif()
# # Setup Python executable
# if(NOT DEFINED Python3_FIND_REGISTRY)
#     # Only consider PATH variable on Windows by default
#     set(Python3_FIND_REGISTRY NEVER)
# endif()
# # Requires Python 3.6+
# find_package(Python3 3.6 COMPONENTS Interpreter Development)
# if(Python3_FOUND)
#     # Setup PYTHON_EXECUTABLE for 3rdparty modules which still use the deprecated find_package(PythonInterp)
#     set(PYTHON_EXECUTABLE
#         ${Python3_EXECUTABLE}
#         CACHE STRING "Deprecated path to the Python executable (for 3rdparty only)" FORCE)
# else()
#     message(FATAL_ERROR "BUILD_PYTHON_MODULE=ON requires Python >= 3.6. Please ensure it is in PATH.")
# endif()
# if(NOT Python3_EXECUTABLE)
#     message(FATAL_ERROR "Python 3 not found in top level file")
# endif()

# We need to get python version to configure some meta files
set(PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")


# execute_process(
#   COMMAND ${Python3_EXECUTABLE} "-c"
#           "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='');"
#   RESULT_VARIABLE PYBIND_CMAKE_DIR_RET
#   OUTPUT_VARIABLE PYBIND_CMAKE_DIR)

# if(PYBIND_CMAKE_DIR_RET MATCHES 0)
#   list(APPEND CMAKE_PREFIX_PATH "${PYBIND_CMAKE_DIR}")
# else()
#   message(ERROR "pybind11 CMake directory not found.")
# endif()

set(PYBIND_MODULE bindings)
message(STATUS "pybind name ${PYBIND_MODULE}")

list(APPEND SRCS src/dg_storage.cpp src/bindings.cpp)
# pybind11_add_module(${PYBIND_MODULE})
pybind11_add_module(bindings src/storage.cpp src/bindings.cpp)

set_target_properties(
    ${PYBIND_MODULE}
    PROPERTIES FOLDER "dgtrt"
               LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

# target_sources(${PYBIND_MODULE} PRIVATE )
include_directories(${CMAKE_SOURCE_DIR}/include)
# set_property(TARGET ${PYBIND_MODULE} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(
  ${PYBIND_MODULE}
  PUBLIC ${Python3_LIBRARIES} )
# target_compile_definitions(${PYBIND_MODULE}
#                            PUBLIC PYBIND_MODULE=${PYBIND_MODULE})

set(PYPI_PACKAGE_NAME "bindings")
set(PYTHON_PACKAGE_SRC_DIR "${PROJECT_SOURCE_DIR}/python")
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")


set(GENERATED_OUTPUTS "")
set(COMPILED_MODULE_PATH_LIST $<TARGET_FILE:bindings> ${CMAKE_SOURCE_DIR}/include)

add_custom_target(
    python-package
    COMMAND
        ${CMAKE_COMMAND} -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR} -DPYTHON_COMPILED_MODULE_DIR=${PYTHON_COMPILED_MODULE_DIR}
        -DPROJECT_ROOT_DIR=${CMAKE_SOURCE_DIR} 
        "-DCOMPILED_MODULE_PATH_LIST=${COMPILED_MODULE_PATH_LIST}" 
        -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/make_python_package.cmake
    VERBATIM
    DEPENDS ${PYBIND_MODULE})

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
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/make_install_pip_package.cmake
    DEPENDS pip-package)

add_custom_target(
    install-python-package
    COMMAND ${Python3_EXECUTABLE} setup.py install --single-version-externally-managed --root=/
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package)
