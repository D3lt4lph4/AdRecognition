// Example : ML assignment data reader 2011
// usage: prog trainingData_file testingData_file

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data ad_cranfield.data"
// and set defintion below to 1 to print out input data

/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <iostream>
#include <fstream>

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 2006 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_VALIDATION_SAMPLES 353  // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define NUMBER_OF_SAMPLES 2006

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
      }
      else if (attribute == ATTRIBUTES_PER_SAMPLE) {
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

void selectNFold(Mat data, Mat classes, Mat &trainingData, Mat &trainingClassifications, Mat &validationData, Mat &validationClassifications, int sampleNumber, int nFolds) {

  int nSamplesPerFold = data.rows / nFolds;

  //copy data for validation sample
  data(Rect(0,nSamplesPerFold * (sampleNumber - 1),ATTRIBUTES_PER_SAMPLE, nSamplesPerFold )).copyTo(validationData);
  classes(Rect(0,nSamplesPerFold * (sampleNumber - 1), NUMBER_OF_CLASSES, nSamplesPerFold )).copyTo(validationClassifications);

  if (sampleNumber == 1) {

    data(Rect(0, nSamplesPerFold, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - 1))).copyTo(trainingData);
    classes(Rect(0,nSamplesPerFold, NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - 1))).copyTo(trainingClassifications);

  } else if (sampleNumber == nFolds) {

    data(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - 1))).copyTo(trainingData);
    classes(Rect(0, 0, NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - 1))).copyTo(trainingClassifications);

  } else {

    trainingData.resize(nSamplesPerFold * (nFolds - 1));
    trainingClassifications.resize(nSamplesPerFold * (nFolds - 1));

    //get first part of the dataset
    data(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (sampleNumber - 1))).copyTo(trainingData(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (sampleNumber - 1))));
    classes(Rect(0, 0, NUMBER_OF_CLASSES, nSamplesPerFold * (sampleNumber - 1))).copyTo(trainingClassifications(Rect(0, 0, NUMBER_OF_CLASSES, nSamplesPerFold * (sampleNumber - 1))));
    //get second part of the dataset
    data(Rect(0, nSamplesPerFold * sampleNumber, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - sampleNumber))).copyTo(trainingData(Rect(0, nSamplesPerFold * (sampleNumber - 1), ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - sampleNumber))));
    classes(Rect(0,nSamplesPerFold * sampleNumber, NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - sampleNumber))).copyTo(trainingClassifications(Rect(0, nSamplesPerFold * (sampleNumber - 1), NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - sampleNumber))));
  }
}

/******************************************************************************/

int main( int argc, char** argv ) {

  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  Mat trainingData = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat trainingClassifications = Mat(NUMBER_OF_TRAINING_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

  //define testing data storage matrices
  Mat validationData = Mat(NUMBER_OF_VALIDATION_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat validationClassifications = Mat(NUMBER_OF_VALIDATION_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

  Mat data = Mat(NUMBER_OF_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat dataClassification = Mat(NUMBER_OF_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);

  //Point for finding the detected class
  Point max_loc = Point(0,0);

  //Creating matrix for validation
  Mat validationSample;
  int correctClass = 0, wrongClass = 0, falsePositives[NUMBER_OF_CLASSES] = {0,0};

  //Vector for mean and std values
  std::vector<float> means, std;

  //Setting the variable for parameter otimization
  int numberOfIterations = 10, paramMin = 10, step = 20, nFolds = 5, bestParam = paramMin;

  //Error to be minimized
  double errorMin = 100;

  Scalar meanError;

  Mat error = Mat(nFolds, numberOfIterations, CV_32FC1);
  //Setting precision for display
  std::cout.precision(4);

  //Creating files to write the data to, will then be used for modelTester (model only, no training)
  std::ofstream csvFile;
  string csvName = "data/csv/", file = "data/models/";

  //Try to read data from the input file
  if (read_data_from_csv(argv[1], data, dataClassification, NUMBER_OF_TRAINING_SAMPLES)) {

    //We look for the best results over all the folds
    for (int fold = 1 ; fold <= nFolds ; fold++) {

      std::cout << "Current fold : " << fold << std::endl;

      //Select the current fold for validation and the rest for training
      selectNFold(data, dataClassification, trainingData, trainingClassifications, validationData, validationClassifications, fold, nFolds);

      //We try to get the best parameter for the model
      for (int i = 0; i < numberOfIterations; i++) {

        //Setting to 0 error variables
        correctClass = 0;
        wrongClass = 0;
        for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
          falsePositives[i] = 0;
        }

        //Setting the layers, the number of neuron will ba what we try to optimize
        int layers_d[] = { ATTRIBUTES_PER_SAMPLE, paramMin + i * step, NUMBER_OF_CLASSES};
        Mat layers = Mat(1,3,CV_32SC1);
        layers.at<int>(0,0) = layers_d[0];
        layers.at<int>(0,1) = layers_d[1];
        layers.at<int>(0,2) = layers_d[2];

        //We create the classifier
        CvANN_MLP* nnetwork = new CvANN_MLP;
        nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

        // set the training parameters
        CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);

        // train the neural network (using training data)
        std::cout << "Training iteration for the classifier : " << i+1 << std::endl;

        int iterations = nnetwork->train(trainingData, trainingClassifications, Mat(), Mat(), params);

        std::cout << "Number of iterations used to train the classifier : " << iterations << std::endl;

        //We test it with the validation data
        std::cout << "Calculating error on the validation sample." << std::endl;

        for (int tsample = 0; tsample < validationData.rows; tsample++) {

          // extract a row from the validation matrix
          validationSample = validationData.row(tsample);

          // run neural network prediction
          nnetwork->predict(validationSample, classificationResult);

          minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

          //getting wrong and correct classification
          if (validationClassifications.at<float>(tsample, max_loc.x) == 1) {
            correctClass++;
          } else {
            wrongClass++;
            falsePositives[(int) max_loc.x]++;
          }
        }

        std::cout << "Result on validation set :\n\tCorrect classification: " << correctClass << ", " << (double)correctClass*100/validationData.rows << "%\n\tWrong classifications: " << wrongClass << " " << (double) wrongClass*100/validationData.rows << "%" << std::endl;

        std::cout << "\tClass nonad false postives : " << (double)falsePositives[1]*100/validationData.rows << std::endl;
        std::cout << "\tClass ad false postives : " << (double)falsePositives[0]*100/validationData.rows << std::endl;

        error.at<float>(fold-1,i) = (double) wrongClass*100/validationData.rows;
      }
    }


    if (argc == 3) {
      //File for the graphic
      csvName.append(argv[2]);
      csvName.append(".csv");
      csvFile.open(csvName, std::ofstream::out | std::ofstream::trunc);
    }

    for (int i = 0; i < numberOfIterations ; i++) {
      meanError = mean(error.col(i));
      if (errorMin > meanError[0]) {
        errorMin = meanError[0];
        bestParam = paramMin + i * step;
      }

      if (argc == 4) {
        csvFile << paramMin + i * step;
        csvFile << ",";
        csvFile << meanError[0];
        csvFile << ";\n";
      }
    }

    std::cout << "The best value for the parameter is : " << bestParam << std::endl;
    std::cout << "The error for this parameter is : " << errorMin << std::endl;

    //If model to save
    if (argc == 3) {
      //File path for model
      file.append(argv[2]);
      file.append(".xml");
      //Calculating the model with the best parameter.
      int layers_d[] = { ATTRIBUTES_PER_SAMPLE, bestParam, NUMBER_OF_CLASSES};
      Mat layers = Mat(1,3,CV_32SC1);
      layers.at<int>(0,0) = layers_d[0];
      layers.at<int>(0,1) = layers_d[1];
      layers.at<int>(0,2) = layers_d[2];

      //We create the classifier
      CvANN_MLP* nnetwork = new CvANN_MLP;
      nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

      // set the training parameters
      CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);

      std::cout << "Training the model with the best parameter on the whole dataset." << std::endl;

      nnetwork->train(trainingData, trainingClassifications, Mat(), Mat(), params);

      nnetwork->save(file.c_str());

      csvFile.close();
    }
    return 0;
  }
  return -1;
}
