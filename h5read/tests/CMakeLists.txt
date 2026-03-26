include(GoogleTest)

execute_process(
    COMMAND dials.data get -q thaumatin_i03_rotation
    OUTPUT_VARIABLE THAUMATIN_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_compile_definitions(THAUMATIN_DATA_PATH="${THAUMATIN_PATH}")

add_executable(test_h5read test_h5read.cc)
target_include_directories(test_h5read PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(test_h5read GTest::gtest_main h5read)

gtest_discover_tests(test_h5read PROPERTIES LABELS h5read-tests)