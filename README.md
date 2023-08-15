# ASD-DiagNet-Confounds

Resting-state functional magnetic resonance imaging (rs-fMRI) is a non-invasive imaging technique widely used in 
neuroscience to understand the functional connectivity of the human brain. While rs- fMRI multi-site data can help 
to understand the inner working of the brain, the data acquisition and processing of this data has many challenges. 
One of the challenges is the variability of the data associated with different acquisitions sites, and different MRI 
machines vendors. Other factors such as population heterogeneity among different sites, with variables such as age and 
gender of the subjects, must also be considered. Given that most of the machine-learning models are developed using these 
rs- fMRI multi-site data sets, the intrinsic confounding effects can adversely affect the generalizability and reliability 
of these computational methods, as well as the imposition of upper limits on the classification scores. 

This repository provides the computational framework to asses the confounding effects on the machine learning analysis 
of static functional connectivity computed from rs-fMRI multi-site data. This computational framework was implemented 
for:  1) the generation of new features from the static functional connectivity computed from the ABIDE rs-fMRI 
multi-site data,  using multiple linear regression models, ComBat harmonization models, and normalization methods; 
and 2) classification of control and patient subjects, using ASD-DiagNet as the machine learning model and the new 
features, as well as baseline and homogeneous sub-samples of the ABIDE data.The results obtained with this computational 
framework are useful for:  i) the identification of the phenotypic and imaging variables producing the confounding effects, 
and ii) to control these confounding effects to maximize the classification scores obtained from the machine learning analysis 
of the rs-fMRI ABIDE multi-site data.

# Prerequisites

This software has been tested on the following dependences:

CUDA version 11.4
Ubuntu version 16.04.6
GPU: NVIDIA Titan Xp

The input data to be used for this computational framework is the preprocessed rs-fMRI data obtained from 17 international 
imaging sites, publicly available in the ABIDE database, with a total of 530 control and 505 autism subjects. 
The preprocessing pipeline chosen for this data was the Configurable Pipeline for the Analysis of Connectomes (CPAC), 
and the filt-global preprocessing strategy. The preprocessing pipeline is described in detail in the ABIDE Preprocessed 
website (http://preprocessed-connectomes-project.org/abide/index.html).

# Install and run the code
1. Download the software
   git clone https://github.com/pcdslab/ASD-DiagNet-Confounds.git
    Please update the data paths.

2. Run the jupyter notebook: sfc_new_features.ipynb to compute the new features, and the  jupyter notebook: ASD-DiagNet-Confounds
   for the classification of control and patients subjects with ASD-DiagNet,  using the new features, the baseline sub-samples and
   the homogeneous sub-samples.

# Publications
If you use the computational framework in this repository please cite our paper:

Oswaldo Artiles, Zeina Al Masry, and Fahad Saeed, 'Confounding Effects on the Performance of Machine Learning Analysis
of Static Functional Connectivity Computed from rs‑fMRI Multi‑site Data', Springer Neuroinformatics, published on line 
15 August 2023. 

# Acknowledgements

This research was funded by National Science Foundation (NSF) award No. TI-2213951. In addition, part of this research 
is funded by supplemental grant to NIH NIGMS R01GM134384. Any opinions, findings, and conclusions or recommendations 
expressed in this material are those of the author (s) and do not necessarily reflect the views of the National Science 
Foundation (NSF) or National Institutes of Health (NIH).







   








