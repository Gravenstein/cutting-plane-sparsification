cmake_minimum_required(VERSION 3.3)
project(Sparsification)
set(CMAKE_CXX_STANDARD 17)
# Without this, we cannot set options using set (don't understand why...)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

add_subdirectory(abseil-cpp)

set(AUTOBUILD ON)
add_subdirectory(scipoptsuite)

add_executable(sparsification src/testmain.cpp src/sparsifier.cc)
target_link_libraries(sparsification libscip)
target_link_libraries(sparsification absl::strings absl::status absl::statusor absl::check absl::flat_hash_map absl::flat_hash_set absl::flags absl::flags_parse)
