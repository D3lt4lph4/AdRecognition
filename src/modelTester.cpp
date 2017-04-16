// Example : ML assignment data reader 2011
// usage: prog training_data_file testing_data_file

// For use with testing/training datasets of the same format as: ad_cranfield.data

// for simple test of data reading run as "reader ad_cranfield.data ad_cranfield.data"
// and set defintion below to 1 to print out input data


/******************************************************************************/

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

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

        if (strcmp(tmps, "ad.") == 0) { // adverts are class 1

          classes.at<float>(line, 0) = 1.0;

        } else if (strcmp(tmps, "nonad.") == 0) { // non adverts are class 2

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

int read_data_from_csv_cnn(const char* filename, Mat data, Mat classes,
  int n_samples )
  {

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

/******************************************************************************/

int main( int argc, char** argv ) {

  //define testing data storage matrices
  Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);


  //Since polymorphisme is overrated (no function predict on the super class)
  CvSVM model1;
  CvRTrees model2;
  CvANN_MLP model3;

  long modelNumber = atoi(argv[3]);

  Mat test_sample;
  int correct_class = 0;
  int wrong_class = 0;
  int false_positives [NUMBER_OF_CLASSES] = {0,0};

  double result;

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  Point max_loc = Point(0,0);

  if (modelNumber == 3) {
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 2, CV_32FC1);
    if (read_data_from_csv_cnn(argv[1], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES)) {

      model3.load(argv[2]);

      printf( "\nCalculating the class for each sample: %s\n\n", argv[2]);

      for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++) {

        // extract a row from the testing matrix
        test_sample = testing_data.row(tsample);
        result = model3.predict(test_sample, classificationResult);
        minMaxLoc(classificationResult, 0, 0, 0, &max_loc);
        if (testing_classifications.at<float>(tsample, max_loc.x) == 1) {
          correct_class++;

        } else {
          wrong_class++;
          false_positives[(int) max_loc.x]++;
        }
      }

      printf( "\nResults on the testing database: %s\n"
      "\tCorrect classification: %d (%g%%)\n"
      "\tWrong classifications: %d (%g%%)\n",
      argv[1],
      correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
      wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

      for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
        false_positives[i],
        (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
      }
      return 0;
    }
  } else {
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);
    if (read_data_from_csv(argv[1], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES)) {
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
        test_sample = testing_data.row(tsample);

        // run the prediction
        switch (modelNumber) {
          case 1:
          result = model1.predict(test_sample);
          if (result == testing_classifications.at<float>(tsample, 0)) {
            correct_class++;
          } else {
            wrong_class++;
            false_positives[(int)(testing_classifications.at<float>(tsample, 0))]++;
          }
          break;
          case 2:
          result = model2.predict(test_sample);
          if (result == testing_classifications.at<float>(tsample, 0)) {
            correct_class++;
          } else {
            wrong_class++;
            false_positives[(int)(testing_classifications.at<float>(tsample, 0))]++;
          }
          break;
          default:
          return -1; //sould never go here;
          break;
        }
      }

      printf( "\nResults on the testing database: %s\n"
      "\tCorrect classification: %d (%g%%)\n"
      "\tWrong classifications: %d (%g%%)\n",
      argv[1],
      correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
      wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

      for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
        false_positives[i],
        (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
      }
      return 0;
    }
  }

  return -1;
}
/******************************************************************************/
