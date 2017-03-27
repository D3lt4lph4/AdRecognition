##########################################################################

# Machine Learning Tools

# Author : Toby Breckon, toby.breckon@cranfield.ac.uk

# Copyright (c) 2009 School of Engineering, Cranfield University
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

##########################################################################

# opencv setup using pkg-config

OPENCV_INCLUDE=`pkg-config opencv --cflags`
OPENCV_LIB=`pkg-config opencv --libs`

# general compiler setup

CC=g++
CC_OPTS=-O2 -Wall

##########################################################################

EXAMPLES=randomize selectlines dt_varimportance typechecker

##########################################################################

all:
	make $(EXAMPLES)

##########################################################################

# Tool 1 - randomize

randomize: randomize.cc $(OBJS) $(HEADERS)
	$(CC) $(CC_OPTS) $(CFLAGS) \
	randomize.cc $(OBJS) -o randomize

##########################################################################

# Tool 2 - selectlines

selectlines: selectlines.cc $(OBJS) $(HEADERS)
	$(CC) $(CC_OPTS) $(CFLAGS) \
	selectlines.cc $(OBJS) -o selectlines

##########################################################################

# Tool 3 - decision tree variable importance

dt_varimportance: dt_varimportance.cpp $(OBJS) $(HEADERS)
	$(CC) $(CC_OPTS) $(CFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) \
	dt_varimportance.cpp $(OBJS) -o dt_varimportance

##########################################################################

# Tool 4 - type checker tool

typechecker: typechecker.cc $(OBJS) $(HEADERS)
	$(CC) $(CC_OPTS) $(CFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) \
	typechecker.cc $(OBJS) -o typechecker

##########################################################################

clean:
	rm -f *~ $(EXAMPLES)

##########################################################################
