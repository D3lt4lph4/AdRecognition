// Example : Neural Natwork classifier
// usage : prog trainingData validationData outputFilesName
// where :
//  - trainingData the training dataset
//  - validationData the validation dataset
//  - outputFilesName the name used for the files output (model and xml), can be ommited

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "neuralNetwork ad_cranfield.data ad_cranfield.data" (need to change some define under)

// Author : Deguerre Benjamin

/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <iostream>
#include <fstream>

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 1606 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_VALIDATION_SAMPLES 400  // ** CHANGE TO YOUR NUMBER OF SAMPLES **

#define NUMBER_OF_CLASSES 2
/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes, int n_samples ) {
  char tmps[10]; // tmp string for reading the "ad." and "nonad." class labels

  // if we can't read the input file then return 0
  FILE* f = fopen( filename, "r" );
  if( !f ) {
    printf("ERROR: cannot read file %s\n",  filename);
    return 0; // all not OK
  }

  // for each sample in the file
  for(int line = 0; line < n_samples; line++) {

    // for each attribute on the line in the file
    for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++) {
      if (attribute < ATTRIBUTES_PER_SAMPLE) {
        // first 256 elements (0-255) in each line are the attributes
        if(fscanf(f, "%f,", &(data.at<float>(line, attribute))) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }

      } else if (attribute == ATTRIBUTES_PER_SAMPLE) {

        if(fscanf(f, "%s.\n", tmps) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }
        if (strcmp(tmps, "ad.") == 0) {
          // adverts are class 1
          classes.at<float>(line, 0) = 1.0;
        }
        else if (strcmp(tmps, "nonad.") == 0) {
          // non adverts are class 2
          classes.at<float>(line, 1) = 1.0;
        }
      }
    }
  }

  fclose(f);
  return 1; // all OK
}

int main( int argc, char** argv ) {

  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  Mat trainingData = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat trainingClassifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 2, CV_32FC1);

  //define testing data storage matrices

  Mat testingData = Mat(NUMBER_OF_VALIDATION_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat testingClassifications = Mat(NUMBER_OF_VALIDATION_SAMPLES, 2, CV_32FC1);

  int numberOfIterations = 10, paramMin = 10, step = 10,bestParam = 0;

  double currentError = 0, errorMin = 0;

  Mat testSample;
  int correctClass = 0;
  int wrongClass = 0;
  int falsePositives [NUMBER_OF_CLASSES] = {0,0};

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  Point max_loc = Point(0,0);

  std::ofstream myfile;
  string csv = "data/csv/";
  if (argc == 4) {
    csv.append(argv[3]);
    csv.append(".csv");
    myfile.open(csv);
  }

  if (read_data_from_csv(argv[1], trainingData, trainingClassifications, NUMBER_OF_TRAINING_SAMPLES) && read_data_from_csv(argv[2], testingData, testingClassifications, NUMBER_OF_VALIDATION_SAMPLES)) {

    for (int i = 0; i < numberOfIterations; i++) {

      correctClass = 0;
      wrongClass = 0;
      for (int j = 0; j < NUMBER_OF_CLASSES; j++) {
        falsePositives[0] = 0;
      }

      int layers_d[] = { ATTRIBUTES_PER_SAMPLE, paramMin + i * step, NUMBER_OF_CLASSES};
      Mat layers = Mat(1,3,CV_32SC1);
      layers.at<int>(0,0) = layers_d[0];
      layers.at<int>(0,1) = layers_d[1];
      layers.at<int>(0,2) = layers_d[2];

      CvANN_MLP* nnetwork = new CvANN_MLP;
      nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

      // set the training parameters
      CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);

      // train the neural network (using training data)
      std::cout << "Training iteration for the classifier : " << i+1 << std::endl;

      int iterations = nnetwork->train(trainingData, trainingClassifications, Mat(), Mat(), params);

      std::cout << "Training iterations for backprop algorithm :" << iterations << std::endl;

      // perform classifier validation and report results
      std::cout << "Calculating error on the validation sample." << std::endl;


      for (int tsample = 0; tsample < NUMBER_OF_VALIDATION_SAMPLES; tsample++) {

        // extract a row from the testing matrix
        testSample = testingData.row(tsample);

        // run neural network prediction
        nnetwork->predict(testSample, classificationResult);

        minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

        if (testingClassifications.at<float>(tsample, max_loc.x) == 1) {
          correctClass++;

        } else {
          wrongClass++;
          falsePositives[(int) max_loc.x]++;
        }
      }

      std::cout << "Result on validation set :\n\tCorrect classification: " << correctClass << ", " << (double)correctClass*100/NUMBER_OF_VALIDATION_SAMPLES << "%\n\tWrong classifications: " << wrongClass << " " << (double) wrongClass*100/NUMBER_OF_VALIDATION_SAMPLES << "%" << std::endl;

      std::cout << "\tClass nonad false postives : " << (double)falsePositives[1]*100/NUMBER_OF_VALIDATION_SAMPLES << std::endl;
      std::cout << "\tClass ad false postives : " << (double)falsePositives[0]*100/NUMBER_OF_VALIDATION_SAMPLES << std::endl;

      if (errorMin == 0) {
        errorMin = (double) wrongClass*100/NUMBER_OF_VALIDATION_SAMPLES;
        currentError = errorMin;
      } else {
        currentError = (double) wrongClass*100/NUMBER_OF_VALIDATION_SAMPLES;
      }

      std::cout << "Previous error minimum :" << errorMin << std::endl;
      std::cout << "Current error : " << currentError << std::endl << std::endl;

      //write to file for display
      if (argc == 4) {
        myfile << paramMin + i * step;
        myfile << ",";
        myfile << currentError;
        myfile << ";\n";
      }

      if (argc == 4 && currentError <= errorMin) {
        string file = "data/models/";
        file.append(argv[3]);
        file.append(".xml");
        nnetwork->save(file.c_str());
        errorMin = currentError;
        bestParam = paramMin + i * step;
      }
    }
    std::cout << "The best value for the parameter is : " << bestParam << std::endl;
    std::cout << "The error for this parameter is : " << errorMin << std::endl;

    if (argc == 4) {
      myfile.close();
    }

    return 0;
  }

  if (argc == 4) {
    myfile.close();
  }
  // not OK : main returns -1

  return -1;
}
/******************************************************************************/
