# Check if we need the span backport, and create/add targets if we do.
#
# Still need to check for USE_SPAN_BACKPORT and include <span> or <span.hpp>,
# and import or define the correct namespace.

include(CheckCXXSymbolExists)

# Check if we have the C++20 span header and symbol. Otherwise, use the backport
check_cxx_symbol_exists(std::span span HAS_CXX20_SPAN)


if(NOT HAS_CXX20_SPAN AND NOT TARGET span)
    # We don't want to build the tcb::span testing targets, so just recreate the interface here
    add_library(span INTERFACE)
    target_sources(span INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../dependencies/span/include/tcb/span.hpp)
    target_include_directories(span INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../dependencies/span/include/tcb)
    target_compile_definitions(span INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:C>>:USE_SPAN_BACKPORT>)
    target_compile_features(span INTERFACE cxx_std_11)

    link_libraries(span)
endif()
