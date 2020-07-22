# Python command to run the code. The code is compatible with python3
PYTHON = python3

# Help
help:
	@echo "---------------HELP-----------------"
	@echo "To test the model type make test"
	@echo "To train the model type make train"
	@echo "------------------------------------"

# This function uses DNN-GP.py to run and test the model
test:
	${PYTHON} DNN-GP.py

# This function uses DNN-GP-training.py to train the model
train:
	${PYTHON} DNN-GP-training.py	

# In this context, the *.project pattern means "anything that has the .project extension"
clean:
	rm -r *.project
