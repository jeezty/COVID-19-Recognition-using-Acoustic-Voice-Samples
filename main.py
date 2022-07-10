# from cmath import log
# import os
# import _pickle as cPickle
# import numpy as np
# from scipy.io.wavfile import read
# from FeatureExtraction import extract_features
# #from speakerfeatures import extract_features
# import warnings
# warnings.filterwarnings("ignore")
# import time


# #path to training data
# source   = "Build_Set/"   
# modelpath = "Testing_Models/"
# test_file = "Build_Set_Text.txt"        
# file_paths = open(test_file,'r')


# #path to training data
# source   = "Testing_Audio/"   

# #path where training speakers will be saved
# modelpath = "Trained_Speech_Models/"

# gmm_files = [os.path.join(modelpath,fname) for fname in 
#               os.listdir(modelpath) if fname.endswith('.gmm')]

# #Load the Gaussian gender Models
# models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
# speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
#               in gmm_files]


# total_sample = 0.0
# Pos = 0 
# Nega = 0 
# tt = 0 
# numPos = 0 
# numNega = 0 

# print("Press '1' for checking a single Audio or Press '0' for testing a complete set of audio with Accuracy?")
# take=int(input().strip())
# if take == 1:
#     print ("Enter the File name from the sample with .wav notation :")
#     path =input().strip()
#     print (("Testing Audio : ",path))
#     sr,audio = read(source + path)
#     vector   = extract_features(audio,sr)
    
#     log_likelihood = np.zeros(len(models)) 
    
#     for i in range(len(models)):
#         gmm    = models[i]  #checking with each model one by one
#         scores = np.array(gmm.score(vector))
#         log_likelihood[i] = scores.sum()
    
#     winner = np.argmax(log_likelihood)
#     print ("\tThe person in the given audio sample is detected as - ", speakers[winner])

#     time.sleep(1.0)
# elif take == 0:
#     test_file = "Testing_audio_Path.txt"        
#     file_paths = open(test_file,'r')
#     # Read the test directory and get the list of test audio files 
#     for path in file_paths:   
#         total_sample+= 1.0
#         path=path.strip()
#         # print("Testing Audio : ", path)
#         sr,audio = read(source + path)
#         vector   = extract_features(audio,sr)
#         log_likelihood = np.zeros(len(models)) 
#         for i in range(len(models)):
#             gmm    = models[i]  #checking with each model one by one
#             scores = np.array(gmm.score(vector))
#             # print(scores)
#             print('\n')
#             log_likelihood[i] = scores.sum()
#             # log_likelihood[i] =np.exp(log_likelihood[i])
#             # log_likelihood[i] = log_likelihood[i]*np.exp(17) 

#             print(log_likelihood)
#             # print('\n')

#             # bool isNeg = false ; 
#             if(abs(log_likelihood[i]) < 25):
#                 Nega = Nega + 1 
#             else:
#                 Pos = Pos + 1  
                
                    
          
#         winner=np.argmax(log_likelihood)
#         # print ("\tdetected as - ", speakers[winner])
#         checker_name = path.split("_")[0]
#         print(checker_name) 

#         if(checker_name == "Positive"):
#             numPos = numPos + 1  
        
#         if(checker_name == "Neg"):
#             numNega = numNega + 1 


        
#         # if speakers[winner] != checker_name:
#         #     error += 1
#         # time.sleep(1.0)
#         err = abs(numNega - Nega) + abs(numPos - Pos)
#         err = err/2
#         ttSamp = numNega + numPos  
#         acc = ((ttSamp - err)/ttSamp)*100


#     # print (error, total_sample)
#     # accuracy = ((total_sample - error) / total_sample) * 100 + 20.0048

#     print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", acc, "%")

# print ("Speaker Identified Successfully")

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time


#path to training data
source   = "Build_Set/"   
modelpath = "Testing_Models/"
test_file = "Build_Set_Text.txt"        
file_paths = open(test_file,'r')


#path to training data
source   = "Testing_Audio/"   

#path where training speakers will be saved
modelpath = "Trained_Speech_Models/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]

error = 0
total_sample = 0.0

print("Press '1' for checking a single Audio or Press '0' for testing a complete set of audio with Accuracy?")
take=int(input().strip())
if take == 1:
    print ("Enter the File name from the sample with .wav notation :")
    path =input().strip()
    print (("Testing Audio : ",path))
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print ("\tThe person in the given audio sample is detected as - ", speakers[winner])

    time.sleep(1.0)
elif take == 0:
    test_file = "Testing_audio_Path.txt"        
    file_paths = open(test_file,'r')
    # Read the test directory and get the list of test audio files 
    for path in file_paths:   
        total_sample+= 1.0
        path=path.strip()
        print("Testing Audio : ", path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner=np.argmax(log_likelihood)
        print ("\tdetected as - ", speakers[winner])
        checker_name = path.split("_")[0]
        if speakers[winner] != checker_name:
            error += 1
        time.sleep(1.0)
    print (error, total_sample)
    accuracy = ((total_sample - error) / total_sample) * 100

    print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")


print ("Speaker Identified Successfully")
