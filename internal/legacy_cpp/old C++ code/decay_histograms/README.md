# Decay histograms

## hadron\_decays\_scan_01.root: ###
* 27 Nov 2019
* histograms:

		const Double_t rap_max_gen = 1.2;
		const Double_t rap_max_det = 0.8;
    	const Double_t pt_min = 0;
    	const Double_t pt_max = 15.;
    	const Int_t n_pt_bins_det = 300;
* comment: only 4 jobs run in parallel on condor. Memory?

## hadron_decays_to_photons_scan_01b.root
* 20 Apr 2020
* only decay photons (decay photon cocktail)

## hadron_decays_to_photons_scan_02.root
* 21 July 2020
* first decay photon scan which uses adjusted PYTHIA decay table
* branching ratios as in ALICE code (PYTHIA6, AliRoot/data/decaytable.dat)

## 