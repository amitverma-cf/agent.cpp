# Helpers for engine plugins built against an installed agent-cpp package (included by
# agent-cppConfig.cmake). See examples/08-engine-plugin-template/ and docs/guides/writing-an-engine.md.
#
#   agent_add_engine(<target> <sources>...)
#       A shared library <target>.so/.dll/.dylib (no "lib" prefix; name it agent-engine-<name>) that
#       includes <agent-cpp/engine_abi.h>, with hidden visibility so only agent_engine_get_v1 is
#       exported. Statically linked dependencies stay private (Linux: --exclude-libs,ALL).
#
#   agent_install_engine(<target> [<runtime libraries>...])
#       Installs the plugin into <prefix>/lib/agent/engines/, where the agent CLI and `agent setup`
#       look, plus runtime libraries it loads through an $ORIGIN RPATH. A symlink whose target is in
#       the list too stays a symlink; any other symlink is replaced by a copy of its target.
#       (Same behaviour as the function of that name inside the agent.cpp build.)

#   agent_deploy_engines(<target> <engine>...)
#       For an application: copies the named engines (llama, audiocpp, sdcpp, trellis, onnx) and the libraries they load into
#       <target dir>/engines/ after every build, and installs them to <bindir>/engines. Engines you do not name are not copied, so
#       they never reach your release. The program finds them there at run time (agent::engines::llama and the like).
#       Also copies agent-engine-host next to the program (for EngineIsolation::Process).

function(agent_add_engine target)
    add_library(${target} SHARED ${ARGN})
    target_link_libraries(${target} PRIVATE agent-cpp::engine_abi)
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
endfunction()

function(agent_install_engine target)
    install(TARGETS ${target}
        LIBRARY DESTINATION lib/agent/engines
        RUNTIME DESTINATION lib/agent/engines
    )
    foreach(_lib ${ARGN})
        get_filename_component(_name "${_lib}" NAME)
        file(REAL_PATH "${_lib}" _real)
        get_filename_component(_real_name "${_real}" NAME)
        get_filename_component(_dir "${_lib}" DIRECTORY)
        if(IS_SYMLINK "${_lib}" AND "${_dir}/${_real_name}" IN_LIST ARGN)
            install(FILES "${_lib}" DESTINATION lib/agent/engines)
        else()
            install(FILES "${_real}" DESTINATION lib/agent/engines RENAME "${_name}")
        endif()
    endforeach()
endfunction()

function(agent_deploy_engines target)
    include(GNUInstallDirs)
    if(NOT agent-cpp_ENGINE_DIR)
        message(FATAL_ERROR "agent_deploy_engines: call find_package(agent-cpp) first")
    endif()
    set(_extra_audiocpp "libaudiocpp*" "audiocpp*.dll")
    set(_extra_onnx "libonnxruntime*" "onnxruntime*.dll")
    foreach(_engine ${ARGN})
        file(GLOB _plugin "${agent-cpp_ENGINE_DIR}/agent-engine-${_engine}.*")
        if(NOT _plugin)
            string(TOUPPER "${_engine}" _u)
            message(FATAL_ERROR "agent_deploy_engines: engine '${_engine}' is not installed in ${agent-cpp_ENGINE_DIR}. "
                                "Build agent.cpp with -DAGENT_ENGINE_${_u}=ON and install it.")
        endif()
        set(_files ${_plugin})
        foreach(_pattern IN LISTS _extra_${_engine})
            file(GLOB _more "${agent-cpp_ENGINE_DIR}/${_pattern}")
            list(APPEND _files ${_more})
        endforeach()
        foreach(_file IN LISTS _files)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/engines"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_file}" "$<TARGET_FILE_DIR:${target}>/engines/"
                VERBATIM)
        endforeach()
        install(FILES ${_files} DESTINATION ${CMAKE_INSTALL_BINDIR}/engines)
    endforeach()
    if(EXISTS "${agent-cpp_ENGINE_HOST}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${agent-cpp_ENGINE_HOST}" "$<TARGET_FILE_DIR:${target}>/"
            VERBATIM)
        install(PROGRAMS "${agent-cpp_ENGINE_HOST}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endfunction()
