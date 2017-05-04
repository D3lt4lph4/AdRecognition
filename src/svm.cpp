// Example : svm classifier
// usage : prog trainingData validationData outputFilesName
// where :
//  - trainingData the training dataset
//  - validationData the validation dataset
//  - outputFilesName the name used for the files output (model), can be ommited

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "svm ad_cranfield.data ad_cranfield.data" (need to change some define under)

// Author : Deguerre Benjamin

/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#include <fstream>

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 1606 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define NUMBER_OF_VALIDATION_SAMPLES 400  // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558

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
        if(fscanf(f, "%s.\n", tmps) !=1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }

        if (strcmp(tmps, "ad.") == 0) {
          // adverts are class 1
          classes.at<float>(line, 0) = 1.0;
        }
        else if (strcmp(tmps, "nonad.") == 0) {
          // non adverts are class 2
          classes.at<float>(line, 0) = 0.0;
        }
      }
      else {
        // store all other data as floating point
        if (fscanf(f, "%f,", &(data.at<float>(line, attribute))) != 1) {
          std::cout << "Error while parsing file" << std::endl;
          return 0;
        }
      }
    }
  }

  fclose(f);
  return 1; // all OK
}

// Calculate the mean value for the column of a matrix
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

// Calculate the std for the column of a matrix
std::vector<float> getSTD(Mat data, std::vector<float> means) {
  std::vector<float> std;
  float sum = 0;

  for (int c = 0; c < data.cols; c++) {
    for (int r = 0; r < data.rows; r++) {
      sum = sum + pow(data.at<float>(r,c) - means.at(c),2);
    }
    std.push_back(sqrt(sum / data.rows));
    sum = 0;
  }
  return std;
}

/******************************************************************************/

int main( int argc, char** argv ) {

  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  Mat trainingData = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat trainingClassifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

  //define validation data storage matrices
  Mat validationData = Mat(NUMBER_OF_VALIDATION_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat validationClassifications = Mat(NUMBER_OF_VALIDATION_SAMPLES, 1, CV_32FC1);

  Mat validationSample;
  int correctClass = 0;
  int wrongClass = 0;
  int falsePositives [NUMBER_OF_CLASSES] = {0,0};
  float result;

  // Vector for mean and std values
  std::vector<float> means, std;

  // Path for data to be saved
  string file = "data/models/", statsName = "data/stats/";

  // File
  std::ofstream statFile;


  Mat priors = Mat(1,2, CV_32FC1);
  priors.at<float>(0,0) = 0.84;
  priors.at<float>(0,1) = 0.16;
  CvMat* pri = new CvMat(priors);

  if (read_data_from_csv(argv[1], trainingData, trainingClassifications, NUMBER_OF_TRAINING_SAMPLES) && read_data_from_csv(argv[2], validationData, validationClassifications, NUMBER_OF_VALIDATION_SAMPLES)) {

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

    CvSVMParams params = CvSVMParams(
    CvSVM::C_SVC,   // Type of SVM, here N classes (see manual)
    CvSVM::SIGMOID,  // kernel type (see manual)
    1.0,			// kernel parameter (degree) for poly kernel only
    1.0,			// kernel parameter (gamma) for poly/rbf kernel only
    0.0,			// kernel parameter (coef0) for poly/sigmoid kernel only
    1,				// SVM optimization parameter C
    0,				// SVM optimization parameter nu (not used for N classe SVM)
    0,				// SVM optimization parameter p (not used for N classe SVM)
    pri,		  	// class wieghts (or priors)
    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100000, 0.001));

    // train SVM classifier (using training data)
    CvSVM* svm = new CvSVM;

    std::cout << "\nTraining the SVM (in progress) ..... " << std::endl;
    std::cout << "(SVM 'grid search' => may take some time!)" << std::endl;

    //Autotrain with k-fold set to 2 (min value)
    svm->train_auto(trainingData, trainingClassifications, Mat(), Mat(), params, 2);

    params = svm->get_params();
    std::cout << "\nUsing optimal parameters degree" << params.degree << ", gamma " << params.gamma << ", ceof0 " << params.coef0 << "\n\t C " << params.C  << ", nu " << params.nu << ", p " << params.p << "\n Training ...... Done\n" << std::endl;
    std::cout << "Number of support vectors for trained SVM = " << svm->get_support_vector_count() << std::endl;

    std::cout << "\nUsing validation database\n" << std::endl;

    for (int tsample = 0; tsample < NUMBER_OF_VALIDATION_SAMPLES; tsample++) {

      // extract a row from the testing matrix
      validationSample = validationData.row(tsample);

      // run SVM classifier
      result = svm->predict(validationSample);

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

    if (argc == 4) {
      file.append(argv[3]);
      file.append(".xml");
      svm->save(file.c_str());

      statsName.append(argv[3]);
      statsName.append(".stats");

      statFile.open(statsName, std::ofstream::out | std::ofstream::trunc);
      statFile << means.size() << " " << std.size();
      for (unsigned int i = 0; i < means.size() ; i++) {
        statFile << " " << means.at(i) ;
      }
      for (unsigned int i = 0; i < std.size(); i++) {
        statFile << " " << std.at(i);
      }
      statFile.close();
    }

    return 0;
  }

  return -1;
}
/******************************************************************************/
