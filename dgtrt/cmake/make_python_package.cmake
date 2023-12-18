
message(STATUS "python package: PYTHON_COMPILED_MODULE_DIR = ${PYTHON_COMPILED_MODULE_DIR}")
set(DGTRT_DIR "${PYTHON_PACKAGE_DST_DIR}/dgtrt")
message(STATUS "DGTRT_DIR = ${DGTRT_DIR}")

# Clean up directory
file(REMOVE_RECURSE ${PYTHON_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${DGTRT_DIR}) # todo: why?

# Create python package. It contains: 1) Pure-python code and misc files, copied from ${PYTHON_PACKAGE_SRC_DIR} 2) The
# compiled python-C++ module, i.e. vega.so (or the equivalents) Optionally other modules e.g. vega_tf_ops.so may be
# included. 3) Configured files and supporting files

# 1) Pure-python code and misc files, copied from ${PYTHON_PACKAGE_SRC_DIR}
file(COPY ${PYTHON_PACKAGE_SRC_DIR}/ DESTINATION ${PYTHON_PACKAGE_DST_DIR}/)

# 2) Configured files and supporting files configure_file("${PYTHON_PACKAGE_SRC_DIR}/" "${PYTHON_PACKAGE_DST_DIR}/")
configure_file("${PYTHON_PACKAGE_DST_DIR}/setup.py" "${PYTHON_PACKAGE_DST_DIR}/setup.py")
configure_file("${PYTHON_PACKAGE_DST_DIR}/dgtrt/__init__.py" "${PYTHON_PACKAGE_DST_DIR}/dgtrt/__init__.py")

# 3) The compiled python-C++ module, which is vegapy pybind so file
foreach(COMPILED_MODULE_PATH ${COMPILED_MODULE_PATH_LIST})
    file(INSTALL ${COMPILED_MODULE_PATH} DESTINATION "${DGTRT_DIR}")
endforeach()
