#include "TPythia8.h"
#include <iostream>

void export_full_pythia_decaytable(const char *out_file = "data/decaytables/decaytable_pythia8_full.dat") {
    TPythia8 tp;
    tp.Pythia8()->particleData.listFF(out_file);
    std::cout << "Wrote full Pythia FF decay table to: " << out_file << std::endl;
}

