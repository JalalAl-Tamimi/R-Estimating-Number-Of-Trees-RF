This script "estimateDensityAnd_ntree.R" provides a calculation of the Density-Based Metric to estimate the density of a dataframe. 
A simulated dataframe is provided with a binomial outcome for classification purpose and 10 predictors. Some predictors are informative (X1, X2 and X8). 
X3 to X7 are non-informative but are correlated with either X1 or X2. This is a normal type of data in phonetics research.
An estimation of the optimal number of trees needed to run a Random Forest (using the "party" package) with the highest accuracy is provided. 

An .RData is also added that contains the dataframe and all results. 

Two methods to check the optimal number of trees for Random Forests are provided: 
* Eyeballing the data for the highest AUC (Area Under the Curve) in the dataframe
* An AUC-based comparison using a non-parametric Z test of significance on correlated ROC (Receiver Operating Curves) curves

These methods are used in the following publications:
* Al-Tamimi, J., (2017). Revisiting Acoustic Correlates of Pharyngealization in Jordanian and Moroccan Arabic: Implications for Formal Representations. Laboratory Phonology: Journal of the Association for Laboratory Phonology, 8(1): 1-40.
* Al-Tamimi, J., and Khattab, G. (under review). Acoustic correlates of the voicing contrast in Lebanese Arabic singleton and geminate plosives. Invited manuscript for the special issue of Journal of Phonetics, “Marking 50 Years of Research on Voice Onset Time and the Voicing Contrast in the World’s Languages" (eds., T. Cho, G. Docherty & D. Whalen).




