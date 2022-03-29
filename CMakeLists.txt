# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14)

project(osqp-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Options.
option(OSQP-CPP_BUILD_TESTS
  "Whether to build tests." ON
)

include(FetchContent)
include(D:/GitHub/vcpkg/scripts/buildsystems/vcpkg.cmake)
find_package(absl CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# OSQP
if (NOT TARGET osqpstatic)
  message(STATUS "osqp-cpp: `osqp` targets not found. Attempting to fetch contents...")
  FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/oxfordcontrol/osqp.git
    GIT_TAG        origin/master
  )
  FetchContent_MakeAvailable(osqp)
else()
  message(STATUS "osqp-cpp: `osqp` targets found.")
endif()

    
message(STATUS "osqp-cpp: Adding osqp-cpp library...")
add_library(osqp-cpp src/osqp++.cc)
target_link_libraries(osqp-cpp PUBLIC absl::strings absl::status absl::statusor Eigen3::Eigen PRIVATE osqpstatic ${CMAKE_DL_LIBS})
target_include_directories(osqp-cpp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
message(STATUS "osqp-cpp: Added osqp-cpp library.")

add_executable(my_first_qp src/my_first_qp.cpp)
target_link_libraries(my_first_qp osqp-cpp)
target_include_directories(my_first_qp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

# Build and register tests.
if (OSQP-CPP_BUILD_TESTS)
  message(STATUS "osqp-cpp: Adding osqp-cpp tests...")
  add_executable(osqp_test test/osqp++_test.cc)
  target_link_libraries(osqp_test absl::status absl::span osqp-cpp GTest::gtest GTest::gmock_main GTest::gtest_main)
  gtest_discover_tests(osqp_test)
  message(STATUS "osqp-cpp: Added osqp-cpp tests.")
endif()

