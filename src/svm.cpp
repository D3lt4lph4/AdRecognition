// Example : ML assignment data reader 2011
// usage: prog trainingData_file testData_file

// For use with testing/training datasets of the same format as:
// ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data
// ad_cranfield.data" and set defintion below to 1 to print out input data

/******************************************************************************/

#include <cv.h>  // opencv general include file
#include <ml.h>  // opencv machine learning include file
#include <fstream>
#include <iostream>

using namespace cv;  // OpenCV API is in the C++ "cv" namespace
using namespace cv::ml;

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)
#define ATTRIBUTES_PER_SAMPLE 1558

#define NUMBER_OF_CLASSES 2
/******************************************************************************/

// Function to load from the csv datafiles.
int read_data_from_csv(const char* filename, Mat &data, Mat &classes) {
  int n_samples = 0;

  std::string line;
  std::ifstream my_file(filename);
  std::string::size_type sz;
  float value;

  // Getting the number of lines.
  while (std::getline(my_file, line))
      ++n_samples;

  my_file.close();
  // iterator might be invalidated from getline()
  my_file.open(filename);

  // Creating the matrices to hold the data.
  data.create(n_samples, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  classes.create(n_samples, 1, CV_32SC1);

  // for each sample in the file
  for (int line_nb = 0; line_nb < n_samples; line_nb++) {
    // for each attribute on the line in the file
    std::getline(my_file, line);

    for (int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1);
         attribute++) {
      if (attribute == ATTRIBUTES_PER_SAMPLE) {
        if (line.compare("ad.") == 0) {
          // adverts are class 1
          classes.at<float>(line_nb, 0) = 1.0;
        } else if (line.compare("nonad.") == 0) {
          // non adverts are class 0
          classes.at<float>(line_nb, 0) = 0.0;
        }
      } else {
        // store all other data as floating point
        value = std::stof(line, &sz);
        // + 1 for the ','
        line = line.substr(sz + 1);
        data.at<float>(line_nb, attribute) = value;
      }
    }
  }
  return 1;  // all OK
}

int main(int argc, char** argv) {
  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  Mat trainingData = Mat();
  Mat trainingClassifications = Mat();

  // define validation data storage matrices
  Mat testData = Mat();
  Mat testClassifications = Mat();

  // Creating matrix for validation
  Mat testSample;
  int correctClass = 0, wrongClass = 0,
      falsePositives[NUMBER_OF_CLASSES] = {0, 0}, kFold = 5;
  float result;
  // Vector for mean and std values
  std::vector<float> means, std;

  // Error to be minimized
  double currentError = 0, errorMin = 0;

  // Setting precision for display
  std::cout.precision(4);

  // Creating files to write the data to, will then be used for modelTester
  // (model only, no training)
  std::ofstream statFile;
  std::string file = "data/models/";

  // If model to save
  if (argc == 4) {
    // File for model
    file.append(argv[3]);
    file.append(".xml");
  }

  // Try to read data from the input file
  if (read_data_from_csv(argv[1], trainingData, trainingClassifications) &&
      read_data_from_csv(argv[2], testData, testClassifications)) {

    Ptr<TrainData> trainData = TrainData::create(trainingData, SampleTypes::ROW_SAMPLE, trainingClassifications);

    // Creating the svm
    Ptr<SVM> svm = SVM::create();

    svm->setKernel(SVM::SIGMOID);
    svm->setDegree(1.0);
    svm->setGamma(1.0);
    svm->setCoef0(1.0);
    svm->setC(1);
    svm->setNu(1);
    svm->setP(1);
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100000, 0.0001));

    std::cout << "\nTraining the SVM (in progress) ..... " << std::endl;
    std::cout << "(SVM 'grid search' => may take some time!)" << std::endl;

    // train using auto training parameter grid search if it is available
    // (i.e. OpenCV 2.x) with 10 fold cross valdiation
    // N.B. this does not search kernel choice
    svm->trainAuto(trainData, kFold);

    // svm->train(trainingData, trainingClassifications, Mat(), Mat(), params);
    std::cout << "\nUsing optimal parameters degree: " << svm->getDegree()
              << ", gamma: " << svm->getGamma() << ", ceof0: " << svm->getCoef0()
              << ", C: " << svm->getC() << ", nu: " << svm->getNu() << ", p: "
              << svm->getP() << "\n Training ...... Done\n"
              << std::endl;
    std::cout << "Number of support vectors for trained SVM = "
              << svm->getSupportVectors().rows << std::endl;

    printf("\nUsing testing database: %s\n\n", argv[2]);

    Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
    for (int tsample = 0; tsample < testData.rows; tsample++) {
      // extract a row from the testing matrix
      testSample = testData.row(tsample);

      // run SVM classifier
      result = svm->predict(testSample, classificationResult);

      if (result == testClassifications.at<float>(tsample, 0)) {
        correctClass++;
      } else {
        wrongClass++;
        falsePositives[(int)result]++;
      }
    }

    std::cout << "Result on test set :\n\tCorrect classification: "
              << correctClass << ", "
              << (double)correctClass * 100 / testData.rows
              << "%\n\tWrong classifications: " << wrongClass << " "
              << (double)wrongClass * 100 / testData.rows << "%"
              << std::endl;

    std::cout << "\tClass nonad false postives : "
              << (double)falsePositives[1] * 100 / testData.rows
              << std::endl;
    std::cout << "\tClass ad false postives : "
              << (double)falsePositives[0] * 100 / testData.rows
              << std::endl;

    if (errorMin == 0) {
      errorMin = (double)wrongClass * 100 / testData.rows;
      currentError = errorMin;
    } else {
      currentError = (double)wrongClass * 100 / testData.rows;
    }

    if (argc == 4) {
      if (currentError <= errorMin) {
        svm->save(file.c_str());
        errorMin = currentError;
      }
    }
    return 0;
  }

  return -1;
}
/******************************************************************************/
