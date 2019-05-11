// Example : ML assignment data reader 2011
// usage: prog trainingData_file validationData_file

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

#define NUMBER_OF_TRAINING_SAMPLES \
  1606  // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_VALIDATION_SAMPLES \
  400  // ** CHANGE TO YOUR NUMBER OF SAMPLES **

#define NUMBER_OF_CLASSES 2
/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes,
                       int n_samples) {
  char tmps[10];  // tmp string for reading the "ad." and "nonad." class labels

  // if we can't read the input file then return 0
  FILE* f = fopen(filename, "r");
  if (!f) {
    printf("ERROR: cannot read file %s\n", filename);
    return 0;  // all not OK
  }

  // for each sample in the file
  for (int line = 0; line < n_samples; line++) {
    // for each attribute on the line in the file
    for (int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1);
         attribute++) {
      if (attribute == ATTRIBUTES_PER_SAMPLE) {
        // last attribute is the class
        if (fscanf(f, "%s.\n", tmps) != 1) {
          std::cout << "Error while parsing file (ATTRIBUTES_PER_SAMPLE) " << filename << std::endl;
          return 0;
        }

        if (strcmp(tmps, "ad.") == 0) {
          // adverts are class 1
          classes.at<float>(line, 0) = 1.0;
        } else if (strcmp(tmps, "nonad.") == 0) {
          // non adverts are class 0
          classes.at<float>(line, 0) = 0.0;
        }
      } else {
        // store all other data as floating point
        if (fscanf(f, "%f,", &(data.at<float>(line, attribute))) != 1) {
          std::cout << "Error while parsing file " << filename  << std::endl;
          return 0;
        }
      }
    }
  }
  fclose(f);
  return 1;  // all OK
}

/******************************************************************************/

int main(int argc, char** argv) {
  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  Mat trainingData =
      Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat trainingClassifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32SC1);

  // define testing data storage matrices
  Mat validationData =
      Mat(NUMBER_OF_VALIDATION_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat validationClassifications =
      Mat(NUMBER_OF_VALIDATION_SAMPLES, 1, CV_32SC1);

  // Creating matrix for validation
  Mat validationSample;
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
  if (read_data_from_csv(argv[1], trainingData, trainingClassifications,
                         NUMBER_OF_TRAINING_SAMPLES) &&
      read_data_from_csv(argv[2], validationData, validationClassifications,
                         NUMBER_OF_VALIDATION_SAMPLES)) {
    
    Ptr<TrainData> trainData = TrainData::create(trainingData, SampleTypes::ROW_SAMPLE, trainingClassifications);
    Ptr<TrainData> valData = TrainData::create(validationData, SampleTypes::ROW_SAMPLE, validationClassifications);


    // train SVM classifier (using training data)

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
    std::cout << "\nUsing optimal parameters degree" << svm->getDegree()
              << ", gamma " << svm->getGamma() << ", ceof0 " << svm->getCoef0()
              << "\n\t C " << svm->getC() << ", nu " << svm->getNu() << ", p "
              << svm->getP() << "\n Training ...... Done\n"
              << std::endl;
    std::cout << "Number of support vectors for trained SVM = "
              << svm->getSupportVectors().rows << std::endl;

    printf("\nUsing testing database: %s\n\n", argv[2]);
    Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
    for (int tsample = 0; tsample < NUMBER_OF_VALIDATION_SAMPLES; tsample++) {
      // extract a row from the testing matrix
      validationSample = validationData.row(tsample);

      // run SVM classifier
      result = svm->predict(validationSample, classificationResult);

      if (result == validationClassifications.at<float>(tsample, 0)) {
        correctClass++;
      } else {
        wrongClass++;
        falsePositives[(int)result]++;
      }
    }

    std::cout << "Result on validation set :\n\tCorrect classification: "
              << correctClass << ", "
              << (double)correctClass * 100 / NUMBER_OF_VALIDATION_SAMPLES
              << "%\n\tWrong classifications: " << wrongClass << " "
              << (double)wrongClass * 100 / NUMBER_OF_VALIDATION_SAMPLES << "%"
              << std::endl;

    std::cout << "\tClass nonad false postives : "
              << (double)falsePositives[1] * 100 / validationData.rows
              << std::endl;
    std::cout << "\tClass ad false postives : "
              << (double)falsePositives[0] * 100 / validationData.rows
              << std::endl;

    if (errorMin == 0) {
      errorMin = (double)wrongClass * 100 / NUMBER_OF_VALIDATION_SAMPLES;
      currentError = errorMin;
    } else {
      currentError = (double)wrongClass * 100 / NUMBER_OF_VALIDATION_SAMPLES;
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
