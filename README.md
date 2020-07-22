# DNN-GP
High throughput wireless access networks, like IEEE 802.11n/ac, have several new physical (PHY) and medium access control (MAC) layer features. The PHY layer enhancements include channel bonding, multiple-input multiple-output (MIMO) antenna technology, advanced modulation and coding schemes (MCS) and short guard interval (SGI). An important enhancement of MAC layer is frame aggregation. These parameters are known as link configuration parameters. However, the optimal combination of these link configuration parameters, which maximizes the network performance, depends on the perceived signal quality of the channel. Moreover, signal quality of a wireless channel changes frequently. The nature of a wireless channel is highly dynamic, nonlinear and time-varying. Therefore, a dynamic mechanism for adaptation of different PHY/MAC parameters is required, which can give a stable and optimized network performance. In this direction, a Deep Neural Network (DNN) based Gaussian Process (GP) regression model can be helpful to predict throughput under a given channel condition and values of several link configuration parameters. We call this learning model as DNN-GP. Specifically, DNN-GP would predict normalized throughput. Later the set of values of link configuration parameters can be chosen, for which we can obtain the maxiumum link layer performance considering the predicted normalized throughput. The DNN-GP model predicts such normalized throughput. So, we need to train the model with a dataset and then we need to test the model. For these two purposes, "DNN-GP-training.py" and "DNN-GP.py" have been used, respectively.
# Dataset:
The dataset contains information regarding normalized throughput achieved under five link configuration parameters and a channel condition measured by SNR. In the dataset, there are seven columns: SNR value, MIMO, channel bandwidth, MCS, guard interval, frame aggregation and normalized throughput. "DNNdata.dat" is the dataset for testing the model. "DNNdatagen.dat" is the dataset which has been used for training the model. The access link of the training dataset is:
http://ieee-dataport.org/documents/ieee-80211ac-performance-dataset
# DNN-GP model:
"DNN-GP-training.py" creates the DNN-GP model. This model takes the dataset (as discussed above) as input for its training purpose. Afte training the model, the trained model is saved in "model-DNNGP.h5".
# Testing model:
"DNN-GP.py" contains the code which needs to be run for testing the model and "model-DNNGP.h5" is the trained model. "rbflayer.py" is a supporting file that is required for executing "DNN-GP.py". This file uses "model-DNNGP.h5" as the trained model to predict normalized throughput.
# Output:
The output of "DNN-GP-training.py" is the trained model and the output of "DNN-GP.py" (testing model) is Normalized Root Mean Square Error (NRMSE) of the predicted normalized throughput. After testing the model, the output would be Normalized Root Mean Square Error (NRMSE) in the prediction of normalized throughput under a channel condition and five link parameter set (mentioned above). NRMSE would be produced after each iteration. 
# Compile and Run:
The code is compatible with python3. The required supporting packages are tensorflow, sklearn, keras and pandas. To compile and run "DNN-GP.py", use the command:  

$ python DNN-GP.py

Using Makefile, the command to test the model is:

$ make test

To train the model, the command is:

$ python DNN-GP-training.py

Using Makefile, the command to train the model is:

$ make train
