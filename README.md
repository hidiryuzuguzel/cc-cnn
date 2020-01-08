# cc-cnn
Learning Color Constancy Using Convolutional Neural Networks (ConvNets)

## Data
Re-processed version of Shi-Gehler (Colour Checker) RAW dataset (https://www2.cs.sfu.ca/~colour/data/shi_gehler/). After downloading the dataset, please update the relevant dataset paths in config file (./assets/shigehler.cfg)

The performance evaluation is done using 3-fold Cross Validation, whose indices are provided in mat file (./assets/gehler_threefoldCVsplit.mat)

**NOTE** that there are multiple Ground Truth for this dataset. I don't use the ground truth provided in the above link, but rather an old version from www.colorconstancy.com, which can be found under (./assets/) folder. 


