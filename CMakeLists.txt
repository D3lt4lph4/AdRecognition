# this file is setup to build everything for the ML examples

cmake_minimum_required (VERSION 2.6)
set( CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}" )

# SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "CMakeCache.txt;CMakeFiles;cmake_install.cmake")

# linux specific stuff

IF ( UNIX )
   set( CMAKE_PREFIX_PATH "/d1/benjamin/opencv/opencv-2.4.13" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
   MESSAGE( "LINUX CONFIG" )
ENDIF ( UNIX )

# windows (inc. 64-bit specific stuff)

IF ( WIN32 )
   set( CMAKE_PREFIX_PATH "C:/OpenCV2.4/build" )
   set( OpenCV_DIR "C:/OpenCV2.4/build" )
   MESSAGE( "WINDOWS CONFIG" )
ENDIF ( WIN32 )

find_package( OpenCV REQUIRED )

project(./bin/svm_test)
add_executable(./bin/svm_test ./src/svm.cpp)
target_link_libraries(./bin/svm_test ${OpenCV_LIBS} )

project(./bin/randomForest_test)
add_executable(./bin/randomForest_test ./src/randomForest.cpp)
target_link_libraries(./bin/randomForest_test ${OpenCV_LIBS} )

project(./bin/neuralNetwork_test)
add_executable(./bin/neuralNetwork_test ./src/neuralNetwork.cpp)
target_link_libraries(./bin/neuralNetwork_test ${OpenCV_LIBS} )

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
