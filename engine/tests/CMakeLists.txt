include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  hivemind_tests
  test_backend_compat.cc
  test_move_gen.cc
  test_tactical_movegen.cc
  test_draw_detection.cc
  test_mate_detection.cc
  test_mate_probe.cc
  test_mate_race.cc
  test_onnx_utils.cc
  test_uci.cc
  test_tournament.cc
  test_node_concurrency.cc
  test_search_memory.cc
  test_root_scan_stop.cc
  test_custom_engine.cc
  test_diagonal_plane.cc
)
target_link_libraries(hivemind_tests PRIVATE hivemind_lib gtest_main)
# Tiny ONNX fixtures (tests/data/make_think_gain_fixtures.py).
target_compile_definitions(hivemind_tests PRIVATE
  HIVEMIND_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(hivemind_tests)
