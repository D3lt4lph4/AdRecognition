#This file might need some change depending on the cmake version used

PROJECT(MLAssigment)

cmake_minimum_required (VERSION 2.6)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set( OpenCV_DIR "/d1/benjamin/opencv/opencv-2.4.13/release")
set( CMAKE_PREFIX_PATH "/d1/benjamin/opencv/opencv-2.4.13" )
set( CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -std=c++11")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")

find_package( OpenCV REQUIRED )

add_executable(svm ./src/svm.cpp)
target_link_libraries(svm ${OpenCV_LIBS} )

add_executable(randomForest ./src/randomForest.cpp)
target_link_libraries(randomForest ${OpenCV_LIBS} )

add_executable(neuralNetwork ./src/neuralNetwork.cpp)
target_link_libraries(neuralNetwork ${OpenCV_LIBS} )

add_executable(randomize ./src/randomize.cpp)
target_link_libraries(randomize ${OpenCV_LIBS} )

add_executable(selectlines ./src/selectlines.cpp)
target_link_libraries(selectlines ${OpenCV_LIBS} )

add_executable(modelTester ./src/modelTester.cpp)
target_link_libraries(modelTester ${OpenCV_LIBS} )

add_executable(selectRatio ./src/selectRatio.cpp)
target_link_libraries(selectRatio ${OpenCV_LIBS} )
