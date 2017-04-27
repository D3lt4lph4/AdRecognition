// Example : ML assignment data reader 2011
// usage: prog training_data_file testingData_file

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data ad_cranfield.data"
// and set defintion below to 1 to print out input data


/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <fstream>

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>


/******************************************************************************/
// global definitions (for speed and ease of use)
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_TESTING_SAMPLES 353  // ** CHANGE TO YOUR NUMBER OF SAMPLES **

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
        if(fscanf(f, "%s.\n", tmps) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        } //valid to be handle

        if (strcmp(tmps, "ad.") == 0) {
          // adverts are class 1
          classes.at<float>(line, 0) = 1.0;

        } else if (strcmp(tmps, "nonad.") == 0) {
          // non adverts are class 0
          classes.at<float>(line, 0) = 0.0;
        }
      } else {
        // store all other data as floating point
        if(fscanf(f, "%f,", &(data.at<float>(line, attribute))) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }//valid to be handle
      }
    }
  }

  fclose(f);
  return 1; // all OK
}

int read_data_from_csv_cnn(const char* filename, Mat data, Mat classes, int n_samples ) {

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
          } else if (strcmp(tmps, "nonad.") == 0) {
            // non adverts are class 2
            classes.at<float>(line, 1) = 1.0;
          }
        }
      }
    }
    fclose(f);
    return 1; // all OK
  }

void readStats(const char* fileName, std::vector<float> &means, std::vector<float> &std) {
  std::fstream stats(fileName, std::ofstream::in);
  int nMeans, nStd;
  float val;

  stats >> nMeans;
  stats >> nStd;
  if (nMeans == nStd) {
    for (int i = 0; i < nMeans; i++) {
      stats >> val;
      means.push_back(val);
    }
    for (int i = 0; i < nStd; i++) {
      stats >> val;
      std.push_back(val);
    }
  } else {
    std::cout << "Error in the data file" << std::endl;
  }
}

void normalizeData(const std::vector<float> means, const std::vector<float> std, Mat data) {
  for (int c = 0; c < data.cols ; c++) {
    for (int r = 0; r < data.rows ; r++) {
      if (std.at(c) != 0) {
        data.at<float>(r,c) = (data.at<float>(r,c) - means.at(c)) / std.at(c);
      }
    }
  }
}
/******************************************************************************/

int main( int argc, char** argv ) {

  //define testing data storage matrices
  Mat testingData = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);

  //Setting precision for the display
  std::cout.precision(4);

  //Since polymorphisme is overrated (no function predict on the super class)
  CvSVM model1;
  CvRTrees model2;
  CvANN_MLP model3;

  long modelNumber = atoi(argv[3]);

  Mat testSample;
  int correctClass = 0;
  int wrongClass = 0;
  int falsePositives [NUMBER_OF_CLASSES] = {0,0};

  double result;

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  Point max_loc = Point(0,0);

  std::vector<float> means, std;

  if (argc == 5) {
    readStats(argv[4], means, std);
  }

  //If Neural Network
  if (modelNumber == 3) {
    Mat testingClassifications = Mat(NUMBER_OF_TESTING_SAMPLES, 2, CV_32FC1);
    if (read_data_from_csv_cnn(argv[1], testingData, testingClassifications, NUMBER_OF_TESTING_SAMPLES)) {

      model3.load(argv[2]);
      if (argc == 5) {
        normalizeData(means, std, testingData);
      }
      std::cout << "\nCalculating the class for each sample\n\n";

      for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++) {

        // extract a row from the testing matrix
        testSample = testingData.row(tsample);
        result = model3.predict(testSample, classificationResult);
        minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

        if (testingClassifications.at<float>(tsample, max_loc.x) == 1) {
          correctClass++;
        } else {
          wrongClass++;
          falsePositives[(int) max_loc.x]++;
        }
      }

      std::cout << "Result on testing set :\n\tCorrect classification: " << correctClass << ", " << (double)correctClass*100/NUMBER_OF_TESTING_SAMPLES << "%\n\tWrong classifications: " << wrongClass << " " << (double) wrongClass*100/NUMBER_OF_TESTING_SAMPLES << "%" << std::endl;

      std::cout << "\tClass nonad false postives : " << (double)falsePositives[1]*100/NUMBER_OF_TESTING_SAMPLES << std::endl;
      std::cout << "\tClass ad false postives : " << (double)falsePositives[0]*100/NUMBER_OF_TESTING_SAMPLES << std::endl;


      return 0;
    }
  } else { //Else svm or Random Forest
    Mat testingClassifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);
    if (read_data_from_csv(argv[1], testingData, testingClassifications, NUMBER_OF_TESTING_SAMPLES)) {
      if (argc == 5) {
        normalizeData(means, std, testingData);
      }
      switch (modelNumber) {
        case 1:
        model1.load(argv[2]);
        break;
        case 2:
        model2.load(argv[2]);
        break;
        default:
        std::cout << "Wrong model." << std::endl;
        return 0;
        break;
      }

      printf( "\nCalculating the class for each sample: %s\n\n", argv[2]);

      for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++) {

        // extract a row from the testing matrix
        testSample = testingData.row(tsample);

        // run the prediction
        switch (modelNumber) {
          case 1:
          result = model1.predict(testSample);
          if (result == testingClassifications.at<float>(tsample, 0)) {
            correctClass++;
          } else {
            wrongClass++;
            falsePositives[(int)result]++;
          }
          break;
          case 2:
          result = model2.predict(testSample);
          if (result == testingClassifications.at<float>(tsample, 0)) {
            correctClass++;
          } else {
            wrongClass++;
            falsePositives[(int)result]++;
          }
          break;
          default:
          return -1; //sould never go here;
          break;
        }
      }

      std::cout << "digit 0 = nonad and 1 = ad" << std::endl;
      std::cout << "Result on testing set :\n\tCorrect classification: " << correctClass << ", " << (double)correctClass*100/NUMBER_OF_TESTING_SAMPLES << "%\n\tWrong classifications: " << wrongClass << " " << (double) wrongClass*100/NUMBER_OF_TESTING_SAMPLES << "%" << std::endl;

      std::cout << "\tClass nonad false postives : " << (double)falsePositives[0]*100/NUMBER_OF_TESTING_SAMPLES << std::endl;
      std::cout << "\tClass ad false postives : " << (double)falsePositives[1]*100/NUMBER_OF_TESTING_SAMPLES << std::endl;
      return 0;
    }
  }

  return -1;
}
