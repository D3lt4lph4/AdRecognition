##########################################################################

# ML Assignment Data Reader

# Author : Toby Breckon, toby.breckon@cranfield.ac.uk

# Copyright (c) 2010 School of Engineering, Cranfield University
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

##########################################################################

# opencv setup using pkg-config

OPENCV_INCLUDE=`pkg-config opencv --cflags`
OPENCV_LIB=`pkg-config opencv --libs`

# general compiler setup

CC=gcc
CC_OPTS=-O2 -Wall

##########################################################################

EXAMPLES=reader

##########################################################################

all:
	make $(EXAMPLES)

##########################################################################

# Example 1 - data reader

reader: reader.cpp $(OBJS) $(HEADERS)
	$(CC) $(CC_OPTS) $(CFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) \
	reader.cpp $(OBJS) -o reader

##########################################################################

clean:
	rm -f *~ $(EXAMPLES)

##########################################################################
