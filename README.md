LD50 predictor ML model made for McGill Pharmahacks Hackathon 2026 using Python language
Used MACCS, Morgan Fingerprints, and molecular descriptors to distinguish the molecules.
The drugs used to train the model came from the Acute Toxicity LD50 dataset from the 
Therapeutics Data Commons (TDC).

The descriptors used to distinguish the molecules came from the following characteristics:

(Topical polar surface area, Wildman-Cripen logP, number of rotatable bonds,number of aromatic rings
Fraction of sp3 Carbons, number of rings, number of non-Hydrogen Carbons, number of rings,
number of hydrogen acceptors (O,N,F), number of hydrogen donors (Lone pairs), 
Net charge of molecules).

These characteristics were derived from RDKit's descriptor functions

The regression model used was XGboost due to the non linear correlation of the data provided

Our model obtained a R^2 coefficient of correlation value of 0.66, Mean absolute error of 0.41
and a root mean squared error of 0.55 for the Test data set. 
