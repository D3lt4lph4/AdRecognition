// Example : ML assignment data reader 2011
// usage: prog trainingData_file validationData_file

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data ad_cranfield.data"
// and set defintion below to 1 to print out input data

/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <iostream>
#include <fstream>

using namespace cv; // OpenCV API is in the C++ "cv" namespace
using namespace cv::ml;

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 2006
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_VALIDATION_SAMPLES 353
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
      if (attribute == ATTRIBUTES_PER_SAMPLE) {

        // last attribute is the class
        if (fscanf(f, "%s.\n", tmps) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }

        if (strcmp(tmps, "ad.") == 0) { // adverts are class 1

          classes.at<float>(line, 0) = 1.0;

        } else if (strcmp(tmps, "nonad.") == 0) { // non adverts are class 2

          classes.at<float>(line, 0) = 0.0;
        }
      } else {
        // store all other data as floating point
        if (fscanf(f, "%f,", &(data.at<float>(line, attribute))) != 1 ) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
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
  classes(Rect(0,nSamplesPerFold * (sampleNumber - 1), 1, nSamplesPerFold )).copyTo(validationClassifications);

  if (sampleNumber == 1) {

    data(Rect(0, nSamplesPerFold, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - 1))).copyTo(trainingData);
    classes(Rect(0,nSamplesPerFold, 1, nSamplesPerFold * (nFolds - 1))).copyTo(trainingClassifications);

  } else if (sampleNumber == nFolds) {

    data(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - 1))).copyTo(trainingData);
    classes(Rect(0, 0, 1, nSamplesPerFold * (nFolds - 1))).copyTo(trainingClassifications);

  } else {

    trainingData.resize(nSamplesPerFold * (nFolds - 1));
    trainingClassifications.resize(nSamplesPerFold * (nFolds - 1));

    //get first part of the dataset
    data(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (sampleNumber - 1))).copyTo(trainingData(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (sampleNumber - 1))));
    classes(Rect(0, 0, 1, nSamplesPerFold * (sampleNumber - 1))).copyTo(trainingClassifications(Rect(0, 0, 1, nSamplesPerFold * (sampleNumber - 1))));
    //get second part of the dataset
    data(Rect(0, nSamplesPerFold * sampleNumber, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - sampleNumber))).copyTo(trainingData(Rect(0, nSamplesPerFold * (sampleNumber - 1), ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - sampleNumber))));
    classes(Rect(0,nSamplesPerFold * sampleNumber, 1, nSamplesPerFold * (nFolds - sampleNumber))).copyTo(trainingClassifications(Rect(0, nSamplesPerFold * (sampleNumber - 1), 1, nSamplesPerFold * (nFolds - sampleNumber))));
  }
}
/******************************************************************************/

int main( int argc, char** argv ) {

  Mat trainingData = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat trainingClassifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

  //define validation data storage matrices
  Mat validationData = Mat(NUMBER_OF_VALIDATION_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat validationClassifications = Mat(NUMBER_OF_VALIDATION_SAMPLES, 1, CV_32FC1);

  Mat data = Mat(NUMBER_OF_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat dataClassification = Mat(NUMBER_OF_SAMPLES, 1, CV_32FC1);

  // define all the attributes as numerical (** not needed for all ML techniques **)
  Mat varType = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
  varType.setTo(Scalar(VAR_NUMERICAL) ); // all inputs are numerical

  varType.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = VAR_CATEGORICAL;

  Mat validationSample;
  int correctClass = 0, wrongClass = 0, falsePositives[NUMBER_OF_CLASSES] = {0,0};

  std::vector<float> means, std;

  int numberOfIterations = 10, paramMin = 100, step = 100, nFolds = 5, bestParam = paramMin;

  double errorMin = 100;

  float result;

  Scalar meanError;

  Mat error = Mat(nFolds, numberOfIterations, CV_32FC1);

  //Setting precision for display
  std::cout.precision(4);

  float priors_array[] = {0.85,0.15};  // weights of each classification for classes
  Mat priors = Mat(1, 2, CV_32FC1, &priors_array);
  //Creating files to write the data to, will then be used for modelTester (model only, no training)
  std::ofstream csvFile;
  std::string csvName = "data/csv/", file = "data/models/";

  if (read_data_from_csv(argv[1], data, dataClassification, NUMBER_OF_TRAINING_SAMPLES)) {

    for (int fold = 1 ; fold <= nFolds ; fold++) {

      std::cout << "Current fold : " << fold << std::endl;

      //Select the current fold for validation and the rest for training
      selectNFold(data, dataClassification, trainingData, trainingClassifications, validationData, validationClassifications, fold, nFolds);

      Ptr<TrainData> trainData = TrainData::create(trainingData, SampleTypes::ROW_SAMPLE, trainingClassifications);
      Ptr<TrainData> valData = TrainData::create(validationData, SampleTypes::ROW_SAMPLE, validationClassifications);

      //We try to get the best parameter for the model
      for (int i = 0; i < numberOfIterations; i++) {

        //Setting to 0 error variables
        correctClass = 0;
        wrongClass = 0;
        for (int j = 0; j < NUMBER_OF_CLASSES; j++) {
          falsePositives[j] = 0;
        }
        
        Ptr<RTrees> rtree = RTrees::create();

        rtree->setMaxDepth(paramMin + i * step);
        rtree->setMinSampleCount(20);
        rtree->setRegressionAccuracy(0);
        rtree->setMaxCategories(15);
        rtree->setPriors(priors);
        rtree->setCalculateVarImportance(false);
        rtree->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-6));

        // train random forest classifier (using training data)
        std::cout << "Training iteration for the classifier : " << i+1 << std::endl;

        
        rtree->train(trainData);

        //We test it with the validation data
        std::cout << "Calculating error on the validation sample." << std::endl;

        for (int tsample = 0; tsample < validationData.rows; tsample++) {
          // extract a row from the testing matrix
          validationSample = validationData.row(tsample);

          // run random forest prediction
          result = rtree->predict(validationSample, Mat());

          // if the prediction and the (true) testing classification are the same
          // (N.B. openCV uses a floating point decision tree implementation!)
          if (result == validationClassifications.at<float>(tsample, 0)) {
            correctClass++;
          } else {
            wrongClass++;
            falsePositives[(int)result]++;
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

      if (argc == 3) {
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
      Ptr<RTrees> rtree = RTrees::create();

      rtree->setMaxDepth(bestParam);
      rtree->setMinSampleCount(20);
      rtree->setRegressionAccuracy(0);
      rtree->setMaxCategories(15);
      rtree->setPriors(priors);
      rtree->setCalculateVarImportance(false);
      rtree->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-6));

      Ptr<TrainData> tData = TrainData::create(data, SampleTypes::ROW_SAMPLE, dataClassification);

      rtree->train(tData);

      rtree->save(file.c_str());

      csvFile.close();
    }
    return 0;
  }
  return -1;
}
