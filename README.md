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



