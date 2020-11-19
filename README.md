# AR-Privacy-KK
Git repository for the python scripts for my AR-Privacy Thesis research.  

### img_to_video.py  
This script takes images in the folder "./Videos Dataset/Images/" according the label and creates a 9 minute video by combining 3 video parts with 6 images each running for 30 seconds. (30 seconds x 6 images x 3 sections = 9 minute video clip)  
The default first section is always the baseline images which are solid colors which allow us to get a baseline where just all the region of interest detectors are running.  
The second section is the section which we know to be the honest function of the app.  
The third section will be variable in an attempt to identify what that the unknown function is in a "malicious" app.  

### preprocess.py
This script takes all the .csv files in the RawData folder, parses them and then places the final output in the ProcessedData folder.  

Flags:\
--raw_data = "Path to raw data folder"  
--op_folder = "Path to output folder (ProcessedData)"  
--scale_data = "Whether to scale the processed data in the dataset to the range (0-1) or not"  
--create_dataset = Include this flag when you want to create the dataset as opposed to simply parsing all CPU values  
--per_segment = Set the number of CPU values included in the dataset per segment (The video is divided into 18 segments)   
--test_size = the percentage of the the total dataset whic hwill be divided into the test set.  

### statistical_classification.py  
This script will take the parsed CPU values and apply statistical techniques like (ks-test and student's t-test) to identify whether a CPU trace comes from a honest app or a malicious app.   

Flags:  
--raw_CPU = 'The path to the file which contains the parsed cpu values for the traces.'    
--labels = 'The path to the csv file containing all the labels for the respective CPU trace'    

### plots.py  
This script will just take 2 files, parse the CPU values and then make plots for the values. There is an option to perform statistical analysis and get the pltos for those results for the 2 traces  

Flags:
--hon_CPU = "Path to the honest app trace csv file"  
--mal_CPU = "Path to the malicious app trace file"  
--stat_example = Include flag to get the statistical analysis (ks-test and t-test for the 3 sections of the CPU values) of the the respective traces.  

### train_test_NN_ARSec.py\
This script will load the dataset placed in the processedData folder be the proprocess.py (flag create_dataset included) script and then train a neural network model on the data. The model structure is loaded from the NN_ARSec_model.py file.\
The trained_model is stored in a folder "./trained_model" relative to the script in a .h5 format.\
The number of epochs required is appended to the name f"trained_model_{num_epochs}.h5".\

Flags:\
--data_path = "Path to data files X_train, y_train, X_test, y_test"\
--model_path = "Output path for trained model where model is placed"\
--num_epochs = Number of epochs the model should be trained for\
--batch_size = The batch size for training the model\

### test_NN_ARSec.py\
This script will just load a trained model from memory and get the confusion matrix for the test set.\

Flags:\
--data_path = "Path to data files X_train, y_train, X_test, y_test"\
--model_path = "Path to the trained_model


### util.py  
This script contains all the functions that are required by the other scripts and does not do anything by itself.  

