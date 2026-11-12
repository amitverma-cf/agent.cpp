# Compiles a vendored ggml's CPU backend without Intel AMX (x86-64, GCC/Clang).
#
# ggml's AMX kernels load the AMX tile configuration once per thread (a static thread_local flag)
# and assume nothing else on that thread changes it. In one process other engines do: ONNX
# Runtime's MLAS reloads its own tile configuration for int8 GEMMs (sherpa, onnx), and every
# ggml-based plugin carries its own ggml copy. Once one of them has run on a thread, ggml's AMX
# matmuls on that thread use the wrong tile shapes and return garbage. Only GGML_NATIVE builds on
# AMX CPUs (Xeon Sapphire Rapids and later) are affected; packaged builds (GGML_NATIVE=OFF) never
# enable AMX. AGENT_GGML_AMX=ON keeps AMX, for setups that run engines out of process
# (Config::engine_isolation = Process).
#
# Two ways in:
#   include(ggml_no_amx) after add_subdirectory() of a ggml project (src/engines/llama), or
#   -DCMAKE_PROJECT_INCLUDE:FILEPATH=<this file> for an ExternalProject (src/engines/sdcpp, ...),
#   where it defers itself to the end of that build's top-level directory.

function(_agent_ggml_no_amx)
    if(NOT TARGET ggml-cpu)
        return()
    endif()
    get_target_property(done ggml-cpu AGENT_GGML_NO_AMX)
    if(done)
        return()
    endif()
    set_target_properties(ggml-cpu PROPERTIES AGENT_GGML_NO_AMX ON)
    target_compile_options(ggml-cpu PRIVATE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16)
endfunction()

if(AGENT_GGML_AMX OR MSVC OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    return()
endif()
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mno-amx-int8 AGENT_HAVE_MNO_AMX)
if(NOT AGENT_HAVE_MNO_AMX)
    return()
endif()
if(TARGET ggml-cpu)
    _agent_ggml_no_amx()
else()
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL _agent_ggml_no_amx)
endif()
