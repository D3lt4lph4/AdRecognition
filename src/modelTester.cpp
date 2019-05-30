// Example : ML assignment data reader 2011
// usage: prog training_data_file testingData_file

// For use with testing/training datasets of the same format as:
// ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data
// ad_cranfield.data" and set defintion below to 1 to print out input data

/******************************************************************************/

#include <cv.h>  // opencv general include file
#include <ml.h>  // opencv machine learning include file
#include <fstream>

using namespace cv;
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

int read_data_from_csv_cnn(const char* filename, Mat data, Mat classes,
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
      if (attribute < ATTRIBUTES_PER_SAMPLE) {
        // first 256 elements (0-255) in each line are the attributes
        if (fscanf(f, "%f,", &(data.at<float>(line, attribute))) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }
      } else if (attribute == ATTRIBUTES_PER_SAMPLE) {
        if (fscanf(f, "%s.\n", tmps) != 1) {
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
  return 1;  // all OK
}

void readStats(const char* fileName, std::vector<float>& means,
               std::vector<float>& std) {
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

void normalizeData(const std::vector<float> means, const std::vector<float> std,
                   Mat data) {
  for (int c = 0; c < data.cols; c++) {
    for (int r = 0; r < data.rows; r++) {
      if (std.at(c) != 0) {
        data.at<float>(r, c) = (data.at<float>(r, c) - means.at(c)) / std.at(c);
      }
    }
  }
}
/******************************************************************************/

int main(int argc, char** argv) {
  // define testing data storage matrices
  Mat testingData =
      Mat();

  // Setting precision for the display
  std::cout.precision(4);

  // Since polymorphisme is overrated (no function predict on the super class)
  Ptr<SVM> model1 = SVM::create();
  Ptr<RTrees> model2 = RTrees::create();
  Ptr<ANN_MLP> model3 = ANN_MLP::create();

  long modelNumber = atoi(argv[3]);

  Mat testSample;
  int correctClass = 0;
  int wrongClass = 0;
  int falsePositives[NUMBER_OF_CLASSES] = {0, 0};

  double result;

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  Point max_loc = Point(0, 0);

  std::vector<float> means, std;

  if (argc == 5) {
    readStats(argv[4], means, std);
  }

  // If Neural Network
  if (modelNumber == 3) {
    Mat testingClassifications = Mat();
    if (read_data_from_csv_cnn(argv[1], testingData, testingClassifications,
                               2)) {
      model3->load(argv[2]);
      if (argc == 5) {
        normalizeData(means, std, testingData);
      }
      std::cout << "\nCalculating the class for each sample\n\n";

      for (int tsample = 0; tsample < 2; tsample++) {
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
                << (double)correctClass * 100 /2 
                << "%\n\tWrong classifications: " << wrongClass << " "
                << (double)wrongClass * 100 / 2<< "%"
                << std::endl;

      std::cout << "\tClass nonad false postives : "
                << (double)falsePositives[1] * 100 /2 
                << std::endl;
      std::cout << "\tClass ad false postives : "
                << (double)falsePositives[0] * 100 /2
                << std::endl;

      return 0;
    }
  } else {  // Else svm or Random Forest
    Mat testingClassifications = Mat();
    if (read_data_from_csv(argv[1], testingData, testingClassifications)) {
      if (argc == 5) {
        normalizeData(means, std, testingData);
      }
      switch (modelNumber) {
        case 1:
          model1->load(argv[2]);
          break;
        case 2:
          model2->load(argv[2]);
          break;
        default:
          std::cout << "Wrong model." << std::endl;
          return 0;
          break;
      }

      printf("\nCalculating the class for each sample: %s\n\n", argv[2]);

      for (int tsample = 0; tsample < 2; tsample++) {
        // extract a row from the testing matrix
        testSample = testingData.row(tsample);

        // run the prediction
        switch (modelNumber) {
          case 1:
            result = model1->predict(testSample);
            if (result == testingClassifications.at<float>(tsample, 0)) {
              correctClass++;
            } else {
              wrongClass++;
              falsePositives[(int)result]++;
            }
            break;
          case 2:
            result = model2->predict(testSample);
            if (result == testingClassifications.at<float>(tsample, 0)) {
              correctClass++;
            } else {
              wrongClass++;
              falsePositives[(int)result]++;
            }
            break;
          default:
            return -1;  // sould never go here;
            break;
        }
      }

      std::cout << "digit 0 = nonad and 1 = ad" << std::endl;
      std::cout << "Result on testing set :\n\tCorrect classification: "
                << correctClass << ", "
                << (double)correctClass * 100 / 2
                << "%\n\tWrong classifications: " << wrongClass << " "
                << (double)wrongClass * 100 / 2<< "%"
                << std::endl;

      std::cout << "\tClass nonad false postives : "
                << (double)falsePositives[0] * 100 / 2
                << std::endl;
      std::cout << "\tClass ad false postives : "
                << (double)falsePositives[1] * 100 / 2
                << std::endl;
      return 0;
    }
  }

  return -1;
}
