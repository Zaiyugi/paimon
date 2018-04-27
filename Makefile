# Paimon: An Eulerian fluid solver with CUDA acceleration
# -------------------------------------------------------
# Copyright (C) 2018 Zachary E. Shore

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or 
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------

.PHONY: all clean
.SUFFIXES: .cpp .h .o .cu .cuh .C

DATE := $(shell date +%F)
UNAME := $(shell uname)

ROOTDIR := .
SRCDIR = src
INCLUDE = include
OBJDIR := obj
LOCAL_LIBDIR = ./lib

HT := /usr
HDSO := /usr/lib64

VPATH = src:include

OFILES = \
	DataObject.o \
	CudaDeleter.o \
	FluidTank.o \
	CFDSolver.o

OBJS = $(patsubst %, $(OBJDIR)/%, $(OFILES))

PAIMONLIB = $(LOCAL_LIBDIR)/libPaimon.a

INC_LOCAL := -I./include -I. 
INC_CUDA = -I/usr/local/cuda/samples/common/inc
INC_PYTHON := -I/usr/include/python2.7 -I/usr/lib/python2.7/config
INC_SDL := -I`sdl2-config --prefix`/include 

LIB_SDL = -lGL `sdl2-config --libs` -lGLEW -lexpat -lSDL2_ttf -lSDL2_image

# Extra gencode flags for CUDA

CXX = /usr/local/cuda/bin/nvcc -ccbin=/usr/bin/g++-5
#CXX = /usr/local/cuda/bin/nvcc -ccbin=/usr/bin/g++
CUDA_GENCODE = -arch=sm_30

CUDA_VERBOS = -Xptxas="-v"
CFLAGS = -g -O2 -std=c++11 --compiler-options='-W -Wall -Wextra'
COMPILE_FLAGS = `sdl2-config --cflags` -g -O3 -std=c++11 -W -Wall -Weffc++ -Wextra -pedantic
OFLAGS = -c $(INC_LOCAL)

CUDALIB = -L/usr/local/cuda/lib64 -lcudart

SWIGLD = $(CXX) -shared
SWIGEXEC = swig

###

all: $(OBJDIR) $(OBJS)
	ar rv $(PAIMONLIB) $(OBJS) $(CUFILES)

$(OBJDIR):
	@if [ ! -d $(OBJDIR) ]; then \
		echo "-----------------------------"; \
		echo "ERROR: Object directory does not exist"; \
		echo "Creating directory: $(OBJDIR)"; \
		echo "-----------------------------"; \
		mkdir $(OBJDIR); \
	fi

$(OBJDIR)/CudaDeleter.o: $(SRCDIR)/CudaDeleter.cu $(INCLUDE)/CudaDeleter.h
	$(CXX) $(CUDA_GENCODE) $(CFLAGS) $(OFLAGS) $< -o $@ 

$(OBJDIR)/DataObject.o: $(SRCDIR)/DataObject.cu $(INCLUDE)/DataObject.h
	$(CXX) $(CUDA_GENCODE) $(CFLAGS) $(OFLAGS) $< -o $@ 

$(OBJDIR)/FluidTank.o: $(SRCDIR)/FluidTank.cu $(INCLUDE)/FluidTank.h
	$(CXX) $(CUDA_GENCODE) $(CFLAGS) $(OFLAGS) $(INC_CUDA) $< -o $@ 

$(OBJDIR)/CFDSolver.o: $(SRCDIR)/CFDSolver.cpp $(INCLUDE)/CFDSolver.h
	g++ $(COMPILE_FLAGS) $(OFLAGS) $(INC_SDL) $< -o $@ 

main: main.cpp
	g++ -std=c++11 -c main.cpp -I./include/. -o main.o
	g++ -std=c++11 main.o -L./lib -lPaimon -L/usr/local/cuda/lib64 -lcudart $(LIB_SDL)

clean:
	-rm $(OBJDIR)/*.o $(LOCAL_LIBDIR)/*.a
