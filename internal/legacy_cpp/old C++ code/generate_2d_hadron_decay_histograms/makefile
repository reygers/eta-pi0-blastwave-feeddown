ROOTLIBDIR      = $(shell root-config --libdir)
ROOTLIBS        = $(shell root-config --libs)
ROOTINCL        = $(shell root-config --cflags)
ROOTLDFLAGS     = $(shell root-config --ldflags)
PLATFORM        = $(shell root-config --platform)
ARCH            = $(shell root-config --arch)

ifeq ($(ARCH),macosxarm64)
	# mac
	CXX = clang++
	CXXOPT = -Wall -pthread -g -Wno-unused-const-variable
	# SOFLAGS = -dynamiclib -single_module -undefined dynamic_lookup
	SOSUFFIX = dylib
else
	# linux
	CXX = g++
	CXXOPT = -Wall -g -Wno-unknown-pragmas
	SOSUFFIX = so
endif

libfeeddown.so: generate_2d_hadron_decay_histograms.cpp
	$(CXX) $(CXXOPT) generate_2d_hadron_decay_histograms.cpp `root-config --cflags --glibs` -I ${PYTHIA8}/include ${PYTHIA8}/lib/libpythia8.$(SOSUFFIX) $(ROOTLIBDIR)/libEGPythia8.so -o generate_2d_hadron_decay_histograms

make clean:
	rm -f generate_2d_hadron_decay_histograms