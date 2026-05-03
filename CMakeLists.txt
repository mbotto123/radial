cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.7.0 REQUIRED)

deal_ii_initialize_cached_variables()
project(radial)
ENABLE_TESTING()

include_directories(include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
