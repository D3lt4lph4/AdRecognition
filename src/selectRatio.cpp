// Example : Create two subsets from an original one according to a specified ratio
// (also removing any empty lines in the file - i.e. no chars apart from "\n")

// usage: prog inputFile outputNameFile trainRatio
// where:
//	- inputfile is the name of the file containing all of the data;
//  - outputNameFile is the name to use for train & test output file ie : <outputNameFile>.train <outputNameFile>.test
// 	- trainRatio the ratio between the number of samples within the train file and the number of sample. 0.6 for 1000 total sample would mean 600 sample for training (could be a bit off if odd number of class1 &| class2)
// If no trainRatio is specified, 0.5 will be used

// Made for use with the provided dataset.

// Author : Deguerre Benjamin

// Copyright (c) 2009 School of Engineering, Cranfield University
// License : GPLv3 - https://www.gnu.org/licenses/gpl-3.0.fr.html

//Made using the selectlines.cpp file from Toby Breckon

/******************************************************************************/

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <iostream>

using namespace std;

#define LINELENGTHMAX 5000 // all file lines less than 5000 chars

/******************************************************************************/

int main( int argc, char** argv )
{

	vector<char *> inputlinesAd, inputlinesNonAd;	// vector of input lines
	vector<char *>::iterator outline; // iterator for above

	char * line = NULL;

	std::regex e (".*nonad\."); // regex for differanciating between ad and non-ad

	//Setting counter for splitting the data
	int lineN = 0, nSample = 0, middle = 0,trainSampleN = 0;
	double trainRatioWanted = 0.5;

	string fileName = argv[2], fTr, fTe;

	fileName = "data/" + fileName;
	fTr = fileName + ".train", fTe = fileName + ".test";
	// open input file
	FILE* fi = fopen( argv[1], "r" );

	if( !fi ){
		printf("ERROR: cannot read input file %s\n",  argv[1]);
		return -1; // all not OK
	}

	// open output file
	FILE *fTrain = fopen(fTr.c_str(), "w");
	if( !fTrain ){
		printf("ERROR: cannot create output file %s\n",  argv[2]);
		fclose(fi);
		return -1; // all not OK
	}

	FILE *fTest = fopen(fTe.c_str(), "w");
	if( !fTest ){
		printf("ERROR: cannot create output file %s\n",  argv[2]);
		fclose(fi);
		fclose(fTrain);
		return -1; // all not OK
	}

	//Ratio wanted by the user
	if (argc >= 4) {
		trainRatioWanted = atof(argv[3]);
	}


	// read in all the lines of the file (allocating fresh memory for each)
	while (!feof(fi)) {
		line = (char *) malloc(LINELENGTHMAX * sizeof(char));
		fscanf(fi, "%[^\n]\n", line);

		if (std::regex_match (line,e)){
			inputlinesNonAd.push_back(line);
		} else {
			inputlinesAd.push_back(line);
		}
		nSample++;
	}

	//Write ad to the correct file to respect the ratio
	middle = inputlinesAd.size() * trainRatioWanted; //Getting the number of ad sample in the training set
	for(outline = inputlinesAd.begin(); outline < inputlinesAd.end(); outline++) {
		if (lineN <= middle) {
			fprintf(fTrain, "%s\n", *outline);
			trainSampleN++;
		} else {
			fprintf(fTest, "%s\n", *outline);
		}
		lineN++;
		free((void *) *outline); // free memory also, vector screwed from that point
	}

	//Write non-ad to the correct file
	middle = inputlinesNonAd.size() * trainRatioWanted; //Getting the number of non-ad sample in the training set
	lineN = 0;
	for(outline = inputlinesNonAd.begin(); outline < inputlinesNonAd.end(); outline++) {
		if (lineN <= middle) {
			trainSampleN++;
			fprintf(fTrain, "%s\n", *outline);
		} else {
			fprintf(fTest, "%s\n", *outline);
		}
		lineN++;
		free((void *) *outline); // free memory also, vector screwed from that point
	}

	std::cout << "Number of train sample : " << trainSampleN << std::endl;
	std::cout << "Number of test sample : " << nSample - trainSampleN << std::endl;

	fclose(fi);
	fclose(fTrain);
	fclose(fTest);

	return 1; // all OK
}
/******************************************************************************/
