CXX := c++
PYTHIA8_CONFIG := $(if $(PYTHIA8),$(PYTHIA8)/bin/pythia8-config,pythia8-config)
CXXFLAGS := $(shell root-config --cflags) $(shell $(PYTHIA8_CONFIG) --cflags) -O2 -std=c++17 -mmacosx-version-min=15.7
LDFLAGS := $(shell root-config --libs) -lEG -lEGPythia8 $(shell $(PYTHIA8_CONFIG) --libs) $(shell pkg-config --libs hdf5) -lhdf5_cpp -mmacosx-version-min=15.7
CPPFLAGS := $(shell pkg-config --cflags hdf5)

all: kernel_factory

kernel_factory: kernel_factory.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f kernel_factory
