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
#define NUMBER_OF_TESTING_SAMPLES 1000  // ** CHANGE TO YOUR NUMBER OF SAMPLES **

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

                    classes.at<float>(line, 1) = 1.0;
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
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

    //define testing data storage matrices

    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

    // define all the attributes as numerical (** not needed for all ML techniques **)

    // Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    // var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
    //
    // // this is a classification problem so reset the last (+1) output
    // // var_type element to CV_VAR_CATEGORICAL (** not needed for all ML techniques **)
    //
    // var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

    Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
    Point max_loc = Point(0,0);
    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {

      // define the parameters for the neural network (MLP)

      // set the network to be 3 layer 256->10->10
      // - one input node per attribute in a sample
      // - 10 hidden nodes
      // - one output node per class

      // note that the OpenCV neural network (MLP) implementation does not
      // support categorical variables explicitly.
      // So, instead of the output class label, we will use
      // a binary vector of {0,0 ... 1,0,0} components (one element by class)
      // for training and therefore, MLP will give us a vector of "probabilities"
      // at the prediction stage - the highest probability can be accepted
      // as the "winning" class label output by the network

      int layers_d[] = { ATTRIBUTES_PER_SAMPLE, 20,  NUMBER_OF_CLASSES};
      Mat layers = Mat(1,3,CV_32SC1);
      layers.at<int>(0,0) = layers_d[0];
      layers.at<int>(0,1) = layers_d[1];
      layers.at<int>(0,2) = layers_d[2];

      // create the network using a sigmoid function with alpha and beta
      // parameters 0.6 and 1 specified respectively (refer to manual)

      CvANN_MLP* nnetwork = new CvANN_MLP;
      nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

      // set the training parameters

      CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(

                                         // terminate the training after either 1000
                                         // iterations or a very small change in the
                                         // network wieghts below the specified value

                                         cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),

                                         // use backpropogation for training

                                         CvANN_MLP_TrainParams::BACKPROP,

                                         // co-efficents for backpropogation training
                                         // (refer to manual)

                                         0.1,
                                         0.1);

      // train the neural network (using training data)

      printf( "\nUsing training database: %s\n", argv[1]);

      int iterations = nnetwork->train(training_data, training_classifications, Mat(), Mat(), params);

      printf( "Training iterations: %i\n\n", iterations);

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

          // run neural network prediction

          nnetwork->predict(test_sample, classificationResult);

          // The NN gives out a vector of probabilities for each class
          // We take the class with the highest "probability"
          // for simplicity (but we really should also check separation
          // of the different "probabilities" in this vector - what if
          // two classes have very similar values ?)

          minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

          printf("Testing Sample %i -> class result (digit %d)\n", tsample, max_loc.x);

          // if the corresponding location in the testing classifications
          // is not "1" (i.e. this is the correct class) then record this

          if (!(testing_classifications.at<float>(tsample, max_loc.x)))
          {
              // if they differ more than floating point error => wrong class

              wrong_class++;

              false_positives[(int) max_loc.x]++;

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

      // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
