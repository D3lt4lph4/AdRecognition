// Example : ML assignment data reader 2011
// usage: prog training_data_file testingData_file

// For use with testing/training datasets of the same format as:
// ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data
// ad_cranfield.data" and set defintion below to 1 to print out input data

/******************************************************************************/

#include <cv.h>
#include <ml.h>
#include <fstream>

using namespace cv;
using namespace cv::ml;

#include <stdio.h>

#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_CLASSES 2

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

int read_data_from_csv_cnn(const char* filename, Mat &data, Mat &classes) {

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

int main(int argc, char** argv) {
  // define testing data storage matrices
  Mat testingData, testingClassifications, testSample;

  // Setting precision for the display
  std::cout.precision(4);

  // Since polymorphisme is overrated (no function predict on the super class)
  Ptr<SVM> model1 = SVM::create();
  Ptr<ANN_MLP> model3;

  long modelNumber = atoi(argv[3]);

  int correctClass = 0;
  int wrongClass = 0;
  int falsePositives[NUMBER_OF_CLASSES] = {0, 0};

  double result;

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  Point max_loc = Point(0, 0);

  // If Neural Network
  if (modelNumber == 3) {
    Mat testingClassifications = Mat();
    if (read_data_from_csv_cnn(argv[1], testingData, testingClassifications)) {

      std::cout << "\nLoading the model " << argv[2] << std::endl;
      model3 = Algorithm::load<ANN_MLP>(argv[2]);

      std::cout << "\nCalculating the class for each sample\n\n";

      for (int tsample = 0; tsample < testingData.rows; tsample++) {
        // extract a row from the testing matrix
        testSample = testingData.row(tsample);
        result = model3->predict(testSample, classificationResult);
        minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

        if (testingClassifications.at<float>(tsample, max_loc.x) == 1) {
          correctClass++;
        } else {
          wrongClass++;
          falsePositives[(int)max_loc.x]++;
        }
      }

      std::cout << "Result on testing set :\n\tCorrect classification: "
                << correctClass << ", "
                << (double)correctClass * 100 / testingData.rows
                << "%\n\tWrong classifications: " << wrongClass << " "
                << (double)wrongClass * 100 / testingData.rows << "%"
                << std::endl;

      std::cout << "\tClass nonad false postives : "
                << (double)falsePositives[1] * 100 / testingData.rows
                << std::endl;
      std::cout << "\tClass ad false postives : "
                << (double)falsePositives[0] * 100 / testingData.rows
                << std::endl;

      return 0;
    }
  } else {  // Else svm or Random Forest
    if (read_data_from_csv(argv[1], testingData, testingClassifications)) {

      model1 = Algorithm::load<SVM>(argv[2]);

      printf("\nCalculating the class for each sample: %s\n\n", argv[2]);

      Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
      for (int tsample = 0; tsample < testingData.rows; tsample++) {
        // extract a row from the testing matrix
        testSample = testingData.row(tsample);

        result = model1->predict(testSample, classificationResult);

        if (result == testingClassifications.at<float>(tsample, 0)) {
          correctClass++;
        } else {
          wrongClass++;
          falsePositives[(int)result]++;
        }

      }

      std::cout << "digit 0 = nonad and 1 = ad" << std::endl;
      std::cout << "Result on testing set :\n\tCorrect classification: "
                << correctClass << ", "
                << (double)correctClass * 100 / testingData.rows
                << "%\n\tWrong classifications: " << wrongClass << " "
                << (double)wrongClass * 100 / testingData.rows << "%"
                << std::endl;

      std::cout << "\tClass nonad false postives : "
                << (double)falsePositives[0] * 100 / testingData.rows
                << std::endl;
      std::cout << "\tClass ad false postives : "
                << (double)falsePositives[1] * 100 / testingData.rows
                << std::endl;
      return 0;
    }
  }

  return -1;
}
