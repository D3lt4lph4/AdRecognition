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

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 2006
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_VALIDATION_SAMPLES 353

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

std::vector<float> getMeans(Mat data) {
  std::vector<float> means;
  float sum = 0;

  for (int c = 0; c < data.cols; c++) {
    for (int r = 0; r < data.rows; r++) {
      sum = sum + data.at<float>(r,c);
    }
    means.push_back(sum / data.rows);
    sum = 0;
  }
  return means;
}

std::vector<float> getSTD(Mat data, std::vector<float> means) {
  std::vector<float> std;
  float sum = 0;

  for (int c = 0; c < data.cols; c++) {
    for (int r = 0; r < data.rows; r++) {
      sum = sum + pow(data.at<float>(r,c) - means.at(c),2);
    }
    // std::cout << sqrt(sum / data.rows) << std::endl;
    std.push_back(sqrt(sum / data.rows));
    sum = 0;
  }
  return std;
}
/******************************************************************************/

int main( int argc, char** argv ) {

  Mat trainingData = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat trainingClassifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

  //define validation data storage matrices
  Mat validationData = Mat(NUMBER_OF_VALIDATION_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat validationClassifications = Mat(NUMBER_OF_VALIDATION_SAMPLES, 1, CV_32FC1);

  // define all the attributes as numerical (** not needed for all ML techniques **)
  Mat varType = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
  varType.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

  varType.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

  Mat validationSample;
  int correctClass = 0, wrongClass = 0, falsePositives[NUMBER_OF_CLASSES] = {0,0};

  std::vector<float> means, std;

  int numberOfIterations = 10, paramMin = 100, step = 100, optimumParam = 0;

  double result, currentError = 0, errorMin = 0;

  //Setting precision for display
  std::cout.precision(4);

  //Creating files to write the data to, will then be used for modelTester (model only, no training)
  std::ofstream csvFile, statFile;
  string csvName = "data/csv/", statsName = "data/stats/", file = "data/models/";

  //If model to save
  if (argc == 4) {
    //File for model
    file.append(argv[3]);
    file.append(".xml");

    csvName.append(argv[3]);
    statsName.append(argv[3]);
    csvName.append(".csv");
    statsName.append(".stats");
  }

  if (read_data_from_csv(argv[1], trainingData, trainingClassifications, NUMBER_OF_TRAINING_SAMPLES) && read_data_from_csv(argv[2], validationData, validationClassifications, NUMBER_OF_VALIDATION_SAMPLES)) {

    //We open the files to save data if requested
    if (argc == 4) {
      csvFile.open(csvName, std::ofstream::out | std::ofstream::trunc);
      statFile.open(statsName, std::ofstream::out | std::ofstream::trunc);
    }

    //We calculate the mean and variance for normalization
    means = getMeans(trainingData);
    std = getSTD(trainingData, means);

    //normalizing the training data
    for (int c = 0; c < trainingData.cols ; c++) {
      for (int r = 0; r < trainingData.rows ; r++) {
        if (std.at(c) != 0) {
          trainingData.at<float>(r,c) = (trainingData.at<float>(r,c) - means.at(c)) / std.at(c);
        }
      }
    }

    //normalizing the testing data
    for (int c = 0; c < validationData.cols ; c++) {
      for (int r = 0; r < validationData.rows ; r++) {
        if (std.at(c) != 0) {
          validationData.at<float>(r,c) = (validationData.at<float>(r,c) - means.at(c)) / std.at(c);
        }
      }
    }

    //Writing the stats to file
    if (argc == 4) {
      statFile << means.size() << " " << std.size();
      for (unsigned int i = 0; i < means.size() ; i++) {
        statFile << " " << means.at(i) ;
      }
      for (unsigned int i = 0; i < std.size(); i++) {
        statFile << " " << std.at(i);
      }
    }

    float priors[] = {0.85,0.15};  // weights of each classification for classes

    for (int i = 0; i < numberOfIterations; i++) {

      correctClass = 0;
      wrongClass = 0;
      for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        falsePositives[i] = 0;
      }

      CvRTParams params = CvRTParams(paramMin + i * step, // max depth
        20, // min sample count
        0, // regression accuracy: N/A here
        false, // compute surrogate split, no missing data
        15, // max number of categories (use sub-optimal algorithm for larger numbers)
        0, // the array of priors
        false,  // calculate variable importance
        0,       // number of variables randomly selected at node and used to find the best split(s).
        100,	 // max number of trees in the forest
        0.01f,				// forrest accuracy
        CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
      );


      // train random forest classifier (using training data)

      std::cout << "Training iteration for the classifier : " << i+1 << std::endl;

      CvRTrees* rtree = new CvRTrees;

      rtree->train(trainingData, CV_ROW_SAMPLE, trainingClassifications, Mat(), Mat(), varType, Mat(), params);

      //We test it with the validation data
      std::cout << "Calculating error on the validation sample." << std::endl;

      for (int tsample = 0; tsample < NUMBER_OF_VALIDATION_SAMPLES; tsample++) {
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

      std::cout << "Result on validation set :\n\tCorrect classification: " << correctClass << ", " << (double)correctClass*100/NUMBER_OF_VALIDATION_SAMPLES << "%\n\tWrong classifications: " << wrongClass << " " << (double) wrongClass*100/NUMBER_OF_VALIDATION_SAMPLES << "%" << std::endl;

      for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        std::cout << "\tClass (digit " << i << ") false postives : " << (double)falsePositives[i]*100/NUMBER_OF_VALIDATION_SAMPLES << std::endl;
      }

      if (errorMin == 0) {
        errorMin = (double) wrongClass*100/NUMBER_OF_VALIDATION_SAMPLES;
        currentError = errorMin;
      } else {
        currentError = (double) wrongClass*100/NUMBER_OF_VALIDATION_SAMPLES;
      }

      if (argc == 4) {
        //write to file for display
        csvFile << paramMin + i * step;
        csvFile << ",";
        csvFile << currentError;
        csvFile << ";\n";

        if (currentError <= errorMin) {
          rtree->save(file.c_str());
          errorMin = currentError;
          optimumParam = paramMin + i * step;
        }
      }
      std::cout << "Optimum parameter so far :" << optimumParam << std::endl;

    }
    if (argc == 4) {
      csvFile.close();
      statFile.close();
    }
    return 0;
  }

  return -1;
}
