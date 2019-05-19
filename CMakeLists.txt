# this file is setup to build everything for the ML examples

cmake_minimum_required (VERSION 2.6)

project(AdRecognition)

set( CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}" )
set (CMAKE_CXX_STANDARD 11)
# SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "CMakeCache.txt;CMakeFiles;cmake_install.cmake")

find_package( OpenCV REQUIRED )

file(MAKE_DIRECTORY ./bin)

project(./bin/svm)
add_executable(./bin/svm ./src/svm.cpp)
target_link_libraries(./bin/svm ${OpenCV_LIBS} )

project(./bin/randomForest)
add_executable(./bin/randomForest ./src/randomForest.cpp)
target_link_libraries(./bin/randomForest ${OpenCV_LIBS} )

project(./bin/neuralNetwork)
add_executable(./bin/neuralNetwork ./src/neuralNetwork.cpp)
target_link_libraries(./bin/neuralNetwork ${OpenCV_LIBS} )

project(./bin/randomize)
add_executable(./bin/randomize ./src/randomize.cpp)
target_link_libraries(./bin/randomize ${OpenCV_LIBS} )

project(./bin/selectlines)
add_executable(./bin/selectlines ./src/selectlines.cpp)
target_link_libraries(./bin/selectlines ${OpenCV_LIBS} )

project(./bin/modelTester)
add_executable(./bin/modelTester ./src/modelTester.cpp)
target_link_libraries(./bin/modelTester ${OpenCV_LIBS} )

project(./bin/selectRatio)
add_executable(./bin/selectRatio ./src/selectRatio.cpp)
target_link_libraries(./bin/selectRatio ${OpenCV_LIBS} )
