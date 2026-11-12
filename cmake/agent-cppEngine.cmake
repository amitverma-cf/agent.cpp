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
