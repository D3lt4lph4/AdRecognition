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

#define NUMBER_OF_TRAINING_SAMPLES 2359 // ** CHANGE TO YOUR NUMBER OF SAMPLES **
#define ATTRIBUTES_PER_SAMPLE 1558
#define NUMBER_OF_TESTING_SAMPLES 2359  // ** CHANGE TO YOUR NUMBER OF SAMPLES **

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

    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {

        //*********************************************************************

        // DO YOUR ML HERE

        //*********************************************************************


        // all matrix memory freed by destructors

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
