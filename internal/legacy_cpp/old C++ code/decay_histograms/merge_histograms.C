#include <TFile.h>
#include <TH1.h>
#include <TKey.h>
#include <TClass.h>
#include <iostream>
#include <set>

void merge_histograms() {

    const char* file1 = "hadron_decays_scan_01_without_p_and_pbar.root";
    const char* file2 = "hadron_decays_scan_01_only_p_and_pbar_star_def.root";
    const char* outputFileName = "hadron_decays_scan_01_star_p_and_pbar_def.root";

    // Open the first input ROOT file
    TFile* inputFile1 = TFile::Open(file1, "READ");
    if (!inputFile1 || inputFile1->IsZombie()) {
        std::cerr << "Error: Cannot open first input file " << file1 << std::endl;
        return;
    }
    
    // Open the second input ROOT file
    TFile* inputFile2 = TFile::Open(file2, "READ");
    if (!inputFile2 || inputFile2->IsZombie()) {
        std::cerr << "Error: Cannot open second input file " << file2 << std::endl;
        inputFile1->Close();
        return;
    }
    
    // Create the output ROOT file
    TFile* outputFile = new TFile(outputFileName, "RECREATE");
    if (!outputFile || outputFile->IsZombie()) {
        std::cerr << "Error: Cannot create output file " << outputFileName << std::endl;
        inputFile1->Close();
        inputFile2->Close();
        return;
    }
    
    std::set<std::string> histogramNames;
    
    // Function to copy histograms from a file
    auto copyHistograms = [&](TFile* inputFile) {
        TIter next(inputFile->GetListOfKeys());
        TKey* key;
        while ((key = (TKey*)next())) {
            if (key->GetClassName() && TClass::GetClass(key->GetClassName())->InheritsFrom(TH1::Class())) {
                TH1* hist = (TH1*)key->ReadObj();
                std::string histName = hist->GetName();
                
                // Only copy if not already copied from file1
                if (histogramNames.find(histName) == histogramNames.end()) {
                    outputFile->cd();
                    hist->Write();
                    histogramNames.insert(histName);
                }
            }
        }
    };
    
    // Copy histograms from the first file
    copyHistograms(inputFile1);
    
    // Copy histograms from the second file (only if not in first file)
    copyHistograms(inputFile2);
    
    // Close files
    outputFile->Close();
    inputFile1->Close();
    inputFile2->Close();
    
    std::cout << "Merged histograms saved to " << outputFileName << std::endl;

}