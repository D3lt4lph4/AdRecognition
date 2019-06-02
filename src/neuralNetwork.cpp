// Example : ML assignment data reader 2011
// usage: prog trainingData_file testingData_file

// For use with testing/training datasets of the same format as: ad_cranfield.data
/******************************************************************************/

#include <cv.h>
#include <ml.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::ml;

#include <stdio.h>

#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_CLASSES 2

int read_data_from_csv(const char* filename, Mat &data, Mat &classes) {

  int n_samples = 0;
  float value;
  std::string line;
  std::ifstream my_file(filename);
  std::string::size_type sz;

  // Getting the number of lines
  while (std::getline(my_file, line))
      ++n_samples;

  my_file.close();
  // iterator might be invalidated from getline()
  my_file.open(filename);

  // Creating the matrices to hold the data.
  data.create(n_samples, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  classes.create(n_samples, NUMBER_OF_CLASSES, CV_32FC1);

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
          classes.at<float>(line_nb, 1) = 0.0;
        } else if (line.compare("nonad.") == 0) {
          // non adverts are class 0
          classes.at<float>(line_nb, 0) = 0.0;
          classes.at<float>(line_nb, 1) = 1.0;
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
  return 1;
}

void selectNFold(Mat data, Mat classes, Mat &trainingData, Mat &trainingClassifications, Mat &validationData, Mat &validationClassifications, int sampleNumber, int nFolds) {

  int nSamplesPerFold = data.rows / nFolds;

  // Create the matrices to hold the data
  validationData.create(nSamplesPerFold, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  validationClassifications.create(nSamplesPerFold, NUMBER_OF_CLASSES, CV_32FC1);

  trainingData.create(nSamplesPerFold * (nFolds - 1), ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  trainingClassifications.create(nSamplesPerFold * (nFolds - 1), NUMBER_OF_CLASSES, CV_32FC1);

  //copy data for validation sample
  data(Rect(0,nSamplesPerFold * (sampleNumber - 1), ATTRIBUTES_PER_SAMPLE, nSamplesPerFold)).copyTo(validationData);
  classes(Rect(0,nSamplesPerFold * (sampleNumber - 1), NUMBER_OF_CLASSES, nSamplesPerFold)).copyTo(validationClassifications);

  if (sampleNumber == 1) {
    data(Rect(0, nSamplesPerFold, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - 1))).copyTo(trainingData);
    classes(Rect(0,nSamplesPerFold, NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - 1))).copyTo(trainingClassifications);

  } else if (sampleNumber == nFolds) {
    data(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - 1))).copyTo(trainingData);
    classes(Rect(0, 0, NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - 1))).copyTo(trainingClassifications);
  } else {

    //get first part of the dataset
    data(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (sampleNumber - 1))).copyTo(trainingData(Rect(0, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (sampleNumber - 1))));
    classes(Rect(0, 0, NUMBER_OF_CLASSES, nSamplesPerFold * (sampleNumber - 1))).copyTo(trainingClassifications(Rect(0, 0, NUMBER_OF_CLASSES, nSamplesPerFold * (sampleNumber - 1))));
    //get second part of the dataset
    data(Rect(0, nSamplesPerFold * sampleNumber, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - sampleNumber))).copyTo(trainingData(Rect(0, nSamplesPerFold * (sampleNumber - 1), ATTRIBUTES_PER_SAMPLE, nSamplesPerFold * (nFolds - sampleNumber))));
    classes(Rect(0,nSamplesPerFold * sampleNumber, NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - sampleNumber))).copyTo(trainingClassifications(Rect(0, nSamplesPerFold * (sampleNumber - 1), NUMBER_OF_CLASSES, nSamplesPerFold * (nFolds - sampleNumber))));
  }
}

int main( int argc, char** argv ) {

  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  Mat trainingData = Mat();
  Mat trainingClassifications = Mat();

  Mat validationData = Mat();
  Mat validationClassifications = Mat();

  Mat testData = Mat();
  Mat testClassifications = Mat();

  Mat data = Mat();
  Mat dataClassification = Mat();

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);

  //Point for finding the detected class
  Point max_loc = Point(0,0);

  //Creating matrix for validation
  Mat validationSample, testSample;
  int correctClass = 0, wrongClass = 0, falsePositives[NUMBER_OF_CLASSES] = {0,0};

  //Setting the variable for parameter optimization
  int numberOfIterations = 10, paramMin = 10, step = 20, nFolds = 5, bestParam = paramMin;

  //Error to be minimized
  double errorMin = 100;

  Scalar meanError;
  Ptr<ANN_MLP> nnetwork;

  Mat error = Mat(nFolds, numberOfIterations, CV_32FC1);
  //Setting precision for display
  std::cout.precision(4);

  //Creating file to write the data to, will then be used for modelTester (model only, no training)
  std::string model_file = "data/models/";

  //Try to read data from the input file
  if (read_data_from_csv(argv[1], data, dataClassification)) {

    //We look for the best results over all the folds
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

        //Setting the layers, the number of neuron will be what we try to optimize
        int layers_d[] = { ATTRIBUTES_PER_SAMPLE, paramMin + i * step, NUMBER_OF_CLASSES};
        Mat layers = Mat(1,3,CV_32SC1);
        layers.at<int>(0,0) = layers_d[0];
        layers.at<int>(0,1) = layers_d[1];
        layers.at<int>(0,2) = layers_d[2];

        //We create the classifier
        nnetwork = ANN_MLP::create();

        nnetwork->setLayerSizes(layers);
        nnetwork->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
        nnetwork->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.0001));
        nnetwork->setTrainMethod(ANN_MLP::BACKPROP, 0.0001);

        // train the neural network (using training data)
        std::cout << "Training iteration for the classifier : " << i + 1 << std::endl;

        int iterations = nnetwork->train(trainData);

        std::cout << "Number of iterations used to train the classifier : " << iterations << std::endl;

        //We test it with the validation data
        std::cout << "Calculating error on the validation sample." << std::endl;

        for (int tsample = 0; tsample < validationData.rows; tsample++) {

          // extract a row from the validation matrix
          validationSample = validationData.row(tsample);

          // run neural network prediction
          nnetwork->predict(validationSample, classificationResult);

          minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

          std::cout << classificationResult << " " << validationClassifications.at<float>(tsample, 0) << " " << validationClassifications.at<float>(tsample, 1)  << std::endl;

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

    for (int i = 0; i < numberOfIterations ; i++) {
      meanError = mean(error.col(i));
      if (errorMin > meanError[0]) {
        errorMin = meanError[0];
        bestParam = paramMin + i * step;
      }
    }

    std::cout << "The best value for the parameter is : " << bestParam << std::endl;
    std::cout << "The error for this parameter is : " << errorMin << std::endl;

    correctClass = 0, wrongClass = 0;
    falsePositives[0] = 0;
    falsePositives[1] = 0;

    read_data_from_csv(argv[2], testData, testClassifications);

    int layers_bi[] = { ATTRIBUTES_PER_SAMPLE, bestParam, NUMBER_OF_CLASSES};
    Mat layers_b = Mat(1,3,CV_32SC1);
    layers_b.at<int>(0,0) = layers_bi[0];
    layers_b.at<int>(0,1) = layers_bi[1];
    layers_b.at<int>(0,2) = layers_bi[2];
    //We create the classifier
    nnetwork = ANN_MLP::create();

    nnetwork->setLayerSizes(layers_b);
    nnetwork->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    nnetwork->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.0001));
    nnetwork->setTrainMethod(ANN_MLP::BACKPROP, 0.0001);
    
    Ptr<TrainData> nData = TrainData::create(testData, SampleTypes::ROW_SAMPLE, testClassifications);

    std::cout << "Training the model with the best parameter on the whole dataset." << std::endl;

    nnetwork->train(nData);

    for (int tsample = 0; tsample < testData.rows; tsample++) {

      // extract a row from the validation matrix
      testSample = testData.row(tsample);

      // run neural network prediction
      nnetwork->predict(testSample, classificationResult);

      minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

      //getting wrong and correct classification
      if (testClassifications.at<float>(tsample, max_loc.x) == 1) {
        correctClass++;
      } else {
        wrongClass++;
        falsePositives[(int) max_loc.x]++;
      }
    }

    std::cout << "Result on test set :\n\tCorrect classification: " << correctClass << ", " << (double)correctClass*100/testData.rows << "%\n\tWrong classifications: " << wrongClass << " " << (double) wrongClass*100/testData.rows << "%" << std::endl;

    std::cout << "\tClass nonad false postives : " << (double)falsePositives[1]*100/testData.rows << std::endl;
    std::cout << "\tClass ad false postives : " << (double)falsePositives[0]*100/testData.rows << std::endl;

    //If model to save
    if (argc == 4) {
      //File path for model
      model_file.append(argv[3]);
      model_file.append(".xml");
      nnetwork->save(model_file.c_str());

    }
    return 0;
  }
  return -1;
}
