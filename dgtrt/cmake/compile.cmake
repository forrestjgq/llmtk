find_program(CCACHE "ccache")
if(CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    if(NEWCUDA OR JETSON_NX)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Werror -fno-strict-aliasing -Wno-deprecated-declarations -Wno-unused-local-typedefs -Wno-parentheses -Wno-catch-value"
)
set(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -DNDEBUG -O3")

message("flags: " ${CMAKE_CXX_FLAGS})
message("build type: " ${CMAKE_BUILD_TYPE})
