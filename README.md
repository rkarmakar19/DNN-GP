# DNN-GP
# Datset:
The dataset contains information regarding normalized throughput achieved under five link configuration parameters and a channel condition measured by SNR. In the dataset, there are seven columns: SNR value, MIMO, channel bandwidth, MCS, guard interval, frame aggregation and normalized throughput. "DNNdata.dat" is the dataset for testing the model.
# Testing model:
"DNN-GP.py" contains the code which needs to be run for testing the model and "model-DNNGP.h5" is the trained model. "rbflayer.py" is a supporting file that is required for executing "DNN-GP.py". This file uses "model-DNNGP.h5" as the trained model to predict normalized throughput.
# Output:
After testing the model, the output would be Normalized Root Mean Square Error (NRMSE) in the prediction of normalized throughput under a channel condition and five link parameter set (mentioned above). NRMSE would be produced after each iteration.
# Compile and Run:
To compile and run "DNN-GP.py", use the command:  

python DNN-GP.py


