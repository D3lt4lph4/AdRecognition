# AdRecognition

This is a small project done as part of the Cranfield course. It aims to test different machine learning algorithms for the same task. The task is to tell if an input data is an ad or not.

## How to use and requirements

The programs were tested with OpenCV 3.4.2. Some modifications are to be applied for it to work with OpenCV 2.

Compiling the project:

```bash
cmake .
make
```

This will generate 7 executable files:

- randomize, used to randomize the data in an input file and write the randomized output to an other file
- selectlines, select a subset of line from the specified file
- selectRatio, slip the specified dataset in two smaller dataset given the specified ratio
- neuralNetwork
- randomForest
- svm
- modelTester

They can be used as shown in the examples below:

```bash
# First let's randomize the data to remove some possible bias
./bin/randomize data/ad_cranfield.data data/ad_cranfieldRandomized.data

# To select a subset of lines
# ./bin/selectlines <firstline> <lastline> <input file> <output file>

# Then we split in two sets, one for training and one for testing
./bin/selectRatio data/ad_cranfieldRandomized.data ad_cranfieldRandomizedSub 0.8

# Now let's train the neural network
./bin/neuralNetwork data/ad_cranfieldRandomizedSub.train data/ad_cranfieldRandomizedSub.test neuralNetwork

# Again with the random forest
./bin/randomForest data/ad_cranfieldRandomizedSub.train data/ad_cranfieldRandomizedSub.test randomForest 

# ... And the SVM (kfold is used so we don't use the SubSub files)
./bin/svm data/ad_cranfieldRandomizedSub.train data/ad_cranfieldRandomizedSub.test svm

# We check the models
./bin/modelTester data/ad_cranfieldRandomizedSub.test data/models/svm.xml
./bin/modelTester data/ad_cranfieldRandomizedSub.test data/models/rf.xml
./bin/modelTester data/ad_cranfieldRandomizedSub.test data/models/nn.xml
```

## Description of the files

The data folders contains different types of files.

The data files:

- ad_cranfield.name file - An explanation of the data
- ad_cranfield.data file - The set of data (CSV file format)
- ad_cranfieldRandomized.data - The randomized dataset
- ad_cranfieldRandomizedSub.train/ad_cranfieldRandomizedSub.test - The randomized dataset split into train and test sets
- ad_cranfieldRandomizedSubSub.train/ad_cranfieldRandomizedSubSub.val - The randomized train set split into train and validation

The models files, train example models.

## Credits

The files were created modifying the example files provided for the ml course at Cranfield University. Check the files for more details.
