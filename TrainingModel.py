import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from scipy.io import wavfile 
import librosa
from sklearn.mixture import GaussianMixture as GMM
from FeatureExtraction import extract_features
#from speakerfeatures import extract_features
import warnings
import mutagen 
from mutagen.wave import WAVE 
warnings.filterwarnings("ignore")

#path to training da
# source   = "development_set/"
source   = "Voice_Samples_Training/"   

#path where training speakers will be saved

# dest = "speaker_models/"
# train_file = "development_set_enroll.txt"

dest = "Trained_Speech_Models/"
train_file = "Voice_Samples_Training_Path.txt"        
file_paths = open(train_file,'r')


count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print (path)
    
    # read the audio
    sr1,audio1 = read(source+path) 
    sr,audio = read(source+path) 
    # audio,sr = librosa.load(source+path)
  
    # contains all the metadata about the wavpack file 
    # sample_rate, data = wavfile.read('alarm.wav') 
      
    len_data = len(audio1)  # holds length of the numpy array
    t = len_data / sr1
 
    if(t <= 0.5):
        continue
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio,sr)
    print(len(vector))
    print('\n')
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    if count == 2:    
        gmm = GMM(n_components = 3 , covariance_type='diag',n_init = 3)
        gmm.fit(features)
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
        features = np.asarray(())
        count = 0
    count = count + 1
