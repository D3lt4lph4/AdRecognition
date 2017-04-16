// Example : ML assignment data reader 2011
// usage: prog training_data_file testing_data_file

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data ad_cranfield.data"
// and set defintion below to 1 to print out input data

#define PRINT_CSV_FILE_INPUTS 0

/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 2006 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
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

  // define training data storage matrices (one for attribute examples, one
  // for classifications)
  std::vector<float> means, std;
  Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

  //define testing data storage matrices

  Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

  // Mat weights = Mat(1,2,CV_32FC1);
  // weights.at<float>(0, 0) = 7500;
  // weights.at<float>(0, 1) = 1;
  // // load training and testing data sets
  // CvMat weightsCV = CvMat(weights);

  if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) && read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES)) {

    means = getMeans(training_data);
    std = getSTD(training_data, means);
    //normalizing the training data
    for (int c = 0; c < training_data.cols ; c++) {
      for (int r = 0; r < training_data.rows ; r++) {
        if (std.at(c) != 0) {
          training_data.at<float>(r,c) = (training_data.at<float>(r,c) - means.at(c)) / std.at(c);
        }
      }
    }

    //normalizing the testing data
    for (int c = 0; c < testing_data.cols ; c++) {
      for (int r = 0; r < testing_data.rows ; r++) {
        if (std.at(c) != 0) {
          testing_data.at<float>(r,c) = (testing_data.at<float>(r,c) - means.at(c)) / std.at(c);
        }
      }
    }

    //Setting priors for svm
    Mat priors = Mat(1,2, CV_32FC1);
    priors.at<float>(0,0) = 0.85;
    priors.at<float>(0,1) = 0.15;
    CvMat* pri = new CvMat(priors);

    CvSVMParams params = CvSVMParams(
    CvSVM::C_SVC,   // Type of SVM, here N classes (see manual)
    CvSVM::SIGMOID,  // kernel type (see manual)
    1.0,			// kernel parameter (degree) for poly kernel only
    1.0,			// kernel parameter (gamma) for poly/rbf kernel only
    1.0,			// kernel parameter (coef0) for poly/sigmoid kernel only
    1,				// SVM optimization parameter C
    0,				// SVM optimization parameter nu (not used for N classe SVM)
    0,				// SVM optimization parameter p (not used for N classe SVM)
    pri,		  	// class wieghts (or priors)
    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.001));

    // train SVM classifier (using training data)

    printf( "\nUsing training database: %s\n\n", argv[1]);
    CvSVM* svm = new CvSVM;

    printf( "\nTraining the SVM (in progress) ..... ");
    fflush(NULL);

    printf( "(SVM 'grid search' => may take some time!)");
    fflush(NULL);

    // train using auto training parameter grid search if it is available
    // (i.e. OpenCV 2.x) with 10 fold cross valdiation
    // N.B. this does not search kernel choice

    svm->train_auto(training_data, training_classifications, Mat(), Mat(), params, 2);
    // svm->train(training_data, training_classifications, Mat(), Mat(), params);
    params = svm->get_params();
    printf( "\nUsing optimal parameters degree %f, gamma %f, ceof0 %f\n\t C %f, nu %f, p %f\n Training ..", params.degree, params.gamma, params.coef0, params.C, params.nu, params.p);
    printf( ".... Done\n");

    // get the number of support vectors used to define the SVM decision boundary
    printf("Number of support vectors for trained SVM = %i\n", svm->get_support_vector_count());

    Mat test_sample;
    int correct_class = 0;
    int wrong_class = 0;
    int false_positives [NUMBER_OF_CLASSES];
    float result;

    // zero the false positive counters in a simple loop
    for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
      false_positives[i] = 0;
    }

    printf( "\nUsing testing database: %s\n\n", argv[2]);
    int test = 0;
    for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++) {
      test++;
      // extract a row from the testing matrix
      test_sample = testing_data.row(tsample);

      // run SVM classifier
      result = svm->predict(test_sample);

      if (result == testing_classifications.at<float>(tsample, 0)) {
        correct_class++;

      } else {
        wrong_class++;
        false_positives[(int)result]++;
      }
    }

    printf( "\nResults on the testing database: %s\n"
    "\tCorrect classification: %d (%g%%)\n"
    "\tWrong classifications: %d (%g%%)\n",
    argv[2],
    correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
    wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

    printf( "\tClass (non-ad) false postives 	%d (%g%%)\n",false_positives[ 0],(double)false_positives[0]*100/NUMBER_OF_TESTING_SAMPLES);
    printf( "\tClass (ad) false postives 	%d (%g%%)\n",false_positives[ 1],(double)false_positives[1]*100/NUMBER_OF_TESTING_SAMPLES);

    // all matrix memory freed by destructors
    if (argc == 4) {
      string file = "models/";
      file.append(argv[3]);
      file.append(".xml");
      std::cout << file << std::endl;
      svm->save(file.c_str());
    }
    return 0;
  }

  return -1;
}
/******************************************************************************/
