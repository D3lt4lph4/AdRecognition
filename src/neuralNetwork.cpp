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

#define NUMBER_OF_TRAINING_SAMPLES 2006 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_TESTING_SAMPLES 353  // ** CHANGE TO YOUR NUMBER OF SAMPLES **

#define NUMBER_OF_CLASSES 2
/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat class1, Mat class2, int n_samples) {
  char tmps[10]; // tmp string for reading the "ad." and "nonad." class labels

  // if we can't read the input file then return 0
  FILE* f = fopen( filename, "r" );
  if( !f ) {
    printf("ERROR: cannot read file %s\n",  filename);
    return 0; // all not OK
  }

  Mat nfoldData = Mat(1, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat nfoldLabel = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  // for each sample in the file
  for(int line = 0; line < n_samples; line++) {

    // for each attribute on the line in the file
    for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++) {
      if (attribute == ATTRIBUTES_PER_SAMPLE) {

        // last attribute is the class
        fscanf(f, "%s.\n", tmps);

        if (strcmp(tmps, "ad.") == 0) {
          // adverts are class 1
          nfoldLabel.at<float>(line/1, 0) = 1.0;
        }
        else if (strcmp(tmps, "nonad.") == 0) {
          // non adverts are class 2
          nfoldLabel.at<float>(line/1, 1) = 1.0;
        }
      } else {

        // store all other data as floating point
        fscanf(f, "%f,", &(nfoldData.at<float>(line/1, attribute)));
      }
    }
  }

  fclose(f);

  return 1; // all OK
}

int createNFold(const char* filename, std::vector<Mat> data, std::vector<Mat> classes, int nSamples, int nFolds) {
  Mat class1, class2;
  int nSamplesPerFold = nSamples / nFolds;
  int nClass1;
  if(read_data_from_csv(filename, class1, class2, nSamples)){
    nClass1 = class1.rows / class2.rows * nSamplesPerFold;
    for (int i = 0 ; i < nFolds ; i++) {
      // push data for class1
      data.push_back(class1(Rect(i*nClass1, 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold)));
      classes.push_back(class1(Rect(i*nClass1, ATTRIBUTES_PER_SAMPLE, NUMBER_OF_CLASSES, nSamplesPerFold)));

      // push data for class 2
      data.push_back(class2(Rect(i*(nSamplesPerFold-nClass1), 0, ATTRIBUTES_PER_SAMPLE, nSamplesPerFold)));
      classes.push_back(class2(Rect(i*nClass1, ATTRIBUTES_PER_SAMPLE, NUMBER_OF_CLASSES, nSamplesPerFold)));
    }
  } else {
    std::cout << "Error while parsing file." << std::endl;
    return 0;
  }
}
/******************************************************************************/

int main( int argc, char** argv ) {

  // define training data storage matrices (one for attribute examples, one
  // for classifications)

  int nfold = 10;

  std::vector<Mat> training_data;
  std::vector<Mat> training_classifications;
  //define testing data storage matrices

  std::vector<Mat> testing_data;
  std::vector<Mat> testing_classifications;

  Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
  Point max_loc = Point(0,0);
  // load training and testing data sets

  if (createNFold(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES, nfold) && createNFold(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES, 1)) {

    for (int i = 1; i <= nfold ; i++) {

    }
    // int layers_d[] = { ATTRIBUTES_PER_SAMPLE, 40, NUMBER_OF_CLASSES};
    // Mat layers = Mat(1,3,CV_32SC1);
    // layers.at<int>(0,0) = layers_d[0];
    // layers.at<int>(0,1) = layers_d[1];
    // layers.at<int>(0,2) = layers_d[2];
    //
    // CvANN_MLP* nnetwork = new CvANN_MLP;
    // nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);
    //
    // // set the training parameters
    // CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10000, 0.000001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
    //
    // // train the neural network (using training data)
    // printf( "\nUsing training database: %s\n", argv[1]);
    //
    // int iterations = nnetwork->train(training_data, training_classifications, Mat(), Mat(), params);
    //
    // printf( "Training iterations: %i\n\n", iterations);
    //
    // // perform classifier testing and report results
    //
    // Mat test_sample;
    // int correct_class = 0;
    // int wrong_class = 0;
    // int false_positives [NUMBER_OF_CLASSES] = {0,0};
    //
    // printf( "\nUsing testing database: %s\n\n", argv[2]);
    //
    // for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++) {
    //
    //   // extract a row from the testing matrix
    //   test_sample = testing_data.row(tsample);
    //
    //   // run neural network prediction
    //   nnetwork->predict(test_sample, classificationResult);
    //
    //   minMaxLoc(classificationResult, 0, 0, 0, &max_loc);
    //
    //   if (testing_classifications.at<float>(tsample, max_loc.x)) {
    //     correct_class++;
    //
    //   } else {
    //     wrong_class++;
    //     false_positives[(int) max_loc.x]++;
    //   }
    // }
    //
    // printf( "\nResults on the testing database: %s\n"
    // "\tCorrect classification: %d (%g%%)\n"
    // "\tWrong classifications: %d (%g%%)\n",
    // argv[2],
    // correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
    // wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);
    //
    // for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
    //   printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
    //   false_positives[i],
    //   (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
    // }
    //
    // if (argc == 4) {
    //   string file = "models/";
    //   file.append(argv[3]);
    //   std::cout << file << std::endl;
    //   nnetwork->save(file.c_str());
    // }
    // // all OK : main returns 0

    return 0;
  }

  // not OK : main returns -1

  return -1;
}
/******************************************************************************/
