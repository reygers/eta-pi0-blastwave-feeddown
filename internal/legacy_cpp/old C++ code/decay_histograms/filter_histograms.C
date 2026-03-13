#include <TFile.h>
#include <TH1.h>
#include <TKey.h>
#include <TClass.h>
#include <iostream>

void filter_histograms() {

    const char* inputFileName = "hadron_decays_scan_01.root";
    const char* outputFileName = "hadron_decays_scan_01_without_p_and_pbar.root";

    // Open the input ROOT file
    TFile* inputFile = TFile::Open(inputFileName, "READ");
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error: Cannot open input file " << inputFileName << std::endl;
        return;
    }
    
    // Create the output ROOT file
    TFile* outputFile = new TFile(outputFileName, "RECREATE");
    if (!outputFile || outputFile->IsZombie()) {
        std::cerr << "Error: Cannot create output file " << outputFileName << std::endl;
        inputFile->Close();
        return;
    }
    
    // Loop over keys in the input file
    TIter next(inputFile->GetListOfKeys());
    TKey* key;
    while ((key = (TKey*)next())) {
        // Check if the object is a histogram
        if (key->GetClassName() && TClass::GetClass(key->GetClassName())->InheritsFrom(TH1::Class())) {
            TH1* hist = (TH1*)key->ReadObj();
            
            // Apply selection criterion (e.g., name contains substring)
            if (!strstr(hist->GetName(), "idsec2212") && !strstr(hist->GetName(), "idsec-2212")) {
                // Write the histogram to the output file
                outputFile->cd();
                hist->Write();
            }
        }
    }
    
    // Close files
    outputFile->Close();
    inputFile->Close();
    
    std::cout << "Filtered histograms saved to " << outputFileName << std::endl;
}
