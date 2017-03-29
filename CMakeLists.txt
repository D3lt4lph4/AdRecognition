# this file is setup to build everything for the ML examples

cmake_minimum_required (VERSION 2.6)
set( CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}" )

# SET_DIRECTORY_PROPERTIES(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "CMakeCache.txt;CMakeFiles;cmake_install.cmake")

# linux specific stuff

IF ( UNIX )
   set( CMAKE_PREFIX_PATH "/opt/opencv-2.4.2" )
   MESSAGE( "LINUX CONFIG" )
ENDIF ( UNIX )

# windows (inc. 64-bit specific stuff)

IF ( WIN32 )
   set( CMAKE_PREFIX_PATH "C:/OpenCV2.4/build" )
   set( OpenCV_DIR "C:/OpenCV2.4/build" )
   MESSAGE( "WINDOWS CONFIG" )
ENDIF ( WIN32 )

find_package( OpenCV REQUIRED )

project(./svm_test)
add_executable(./svm_test ./src/svm.cpp)
target_link_libraries( ./svm_test ${OpenCV_LIBS} )

project(./randomForest_test)
add_executable(./randomForest_test ./src/randomForest.cpp)
target_link_libraries( ./randomForest_test ${OpenCV_LIBS} )

project(./neuralNetwork_test)
add_executable(./neuralNetwork_test ./src/neuralNetwork.cpp)
target_link_libraries( ./neuralNetwork_test ${OpenCV_LIBS} )

project(./randomize)
add_executable(./randomize ./src/randomize.cpp)
target_link_libraries( ./randomize ${OpenCV_LIBS} )

project(./selectlines)
add_executable(./selectlines ./src/selectlines.cpp)
target_link_libraries( ./selectlines ${OpenCV_LIBS} )
