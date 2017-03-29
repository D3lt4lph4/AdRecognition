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

#define NUMBER_OF_TRAINING_SAMPLES 1359 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_TESTING_SAMPLES 1000 // ** CHANGE TO YOUR NUMBER OF SAMPLES **

#define NUMBER_OF_CLASSES 2
/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes, int n_samples )
{
  char tmps[10]; // tmp string for reading the "ad." and "nonad." class labels

  // if we can't read the input file then return 0
  FILE* f = fopen( filename, "r" );
  if( !f )
  {
    printf("ERROR: cannot read file %s\n",  filename);
    return 0; // all not OK
  }

  // for each sample in the file

  for(int line = 0; line < n_samples; line++)
  {

    // for each attribute on the line in the file

    for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
    {
      if (attribute == ATTRIBUTES_PER_SAMPLE)
      {

        // last attribute is the class

        fscanf(f, "%s.\n", tmps);

        if (strcmp(tmps, "ad.") == 0)
        {
          // adverts are class 1

          classes.at<float>(line, 0) = 1.0;
        }
        else if (strcmp(tmps, "nonad.") == 0)
        {
          // non adverts are class 2

          classes.at<float>(line, 0) = 0.0;
        }
        #if PRINT_CSV_FILE_INPUTS
        printf("%s\n", tmps);
        #endif
      }
      else
      {

        // store all other data as floating point

        fscanf(f, "%f,", &(data.at<float>(line, attribute)));
        #if PRINT_CSV_FILE_INPUTS
        printf("%f,", data.at<float>(line, attribute));
        #endif
      }
    }
  }

  fclose(f);

  return 1; // all OK
}

/******************************************************************************/

int main( int argc, char** argv )
{

  // define training data storage matrices (one for attribute examples, one
  // for classifications)

  Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

  //define testing data storage matrices

  Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
  Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

  // define all the attributes as numerical (** not needed for all ML techniques **)

  Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
  var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

  // this is a classification problem so reset the last (+1) output
  // var_type element to CV_VAR_CATEGORICAL (** not needed for all ML techniques **)

  var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

  double result;
  // load training and testing data sets

  if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
  read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
  {

    float priors[] = {1,1};  // weights of each classification for classes
    // (all equal as equal samples of each digit)

    CvRTParams params = CvRTParams(25, // max depth
      5, // min sample count
      0, // regression accuracy: N/A here
      false, // compute surrogate split, no missing data
      15, // max number of categories (use sub-optimal algorithm for larger numbers)
      priors, // the array of priors
      false,  // calculate variable importance
      4,       // number of variables randomly selected at node and used to find the best split(s).
      100,	 // max number of trees in the forest
      0.01f,				// forrest accuracy
      CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
    );

    // train random forest classifier (using training data)

    printf( "\nUsing training database: %s\n\n", argv[1]);
    CvRTrees* rtree = new CvRTrees;

    rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
      Mat(), Mat(), var_type, Mat(), params);

      // perform classifier testing and report results

      Mat test_sample;
      int correct_class = 0;
      int wrong_class = 0;
      int false_positives [NUMBER_OF_CLASSES] = {0,0};

      printf( "\nUsing testing database: %s\n\n", argv[2]);

      for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
      {

        // extract a row from the testing matrix

        test_sample = testing_data.row(tsample);

        // run random forest prediction

        result = rtree->predict(test_sample, Mat());

        printf("Testing Sample %i -> class result (digit %d)\n", tsample, (int) result);

        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point decision tree implementation!)

        if (fabs(result - testing_classifications.at<float>(tsample, 0))
        >= FLT_EPSILON)
        {
          // if they differ more than floating point error => wrong class

          wrong_class++;

          false_positives[(int) result]++;

        }
        else
        {

          // otherwise correct

          correct_class++;
        }
      }

      printf( "\nResults on the testing database: %s\n"
      "\tCorrect classification: %d (%g%%)\n"
      "\tWrong classifications: %d (%g%%)\n",
      argv[2],
      correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
      wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

      for (int i = 0; i < NUMBER_OF_CLASSES; i++)
      {
        printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
        false_positives[i],
        (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
      }




      // all matrix memory freed by destructors

      // all OK : main returns 0

      return 0;
    }

    // not OK : main returns -1

    return -1;
  }
  /******************************************************************************/
