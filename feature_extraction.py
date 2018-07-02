import sys
import os
import numpy as np
import pandas as pd
import librosa


class dataset:

    def __init__(self, file_location):

        self.file_location = file_location

        #initializes the dataset
        self.entire_dataset = np.zeros((1, 49))

        #loops through different genres in the file
        for genre_name in os.listdir(file_location):

            genre_name_path = os.path.join(file_location, genre_name)
            for audio_file in os.listdir(genre_name_path):

                if audio_file.endswith(".wav"):
                    genre_array = np.array([genre_name])

                    #y represents the time series and sr the sampling rate
                    y, sr = librosa.load(os.path.join(genre_name_path, audio_file))

                    hop_length = 512

                    #Extracting different timbral features for each frame- Each an array of shape (X, total_no_of_frames)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    flatness = librosa.feature.spectral_flatness(y=y)
                    mfcc_delta = librosa.feature.delta(mfcc)
                    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

                    features_song_entire = np.row_stack((mfcc, contrast, centroid, bandwidth, rolloff, flatness))
                    #features_song_entire is an array of size (24,total_no_of_frames)

                    total_no_of_frames = feature_song_entire.shape[1]

                    #converting the number of frames to a factor of 30
                    for i in range(30):
                        if total_no_of_frames % 30 == 0:
                            break
                        else:
                            total_no_of_frames = total_no_of_frames - 1

                    feature_song_entire = feature_song_entire[:, : total_no_of_frames]
                    one_second_features = np.hsplit(feature_song_entire, 30)
                    #splitting the song into 1 second intervals


                    for one_second in one_second_features:
                        #Taking the mean and max of different features in the frames of the 1 second intervals
                        mean_one_second = one_second.mean(axis=1)
                        max_one_second = one_second.max(axis=1)

                        #all_one_second gives the features and the label
                        all_one_second = np.concatenate((mean_one_second, max_one_second, genre_array))


                        self.entire_dataset = np.row_stack((self.entire_dataset, all_one_second))

    def get_dataset(self):
        #The initialized value of zeros is removed
        self.entire_dataset = self.entire_dataset[1:, :]
        return self.entire_dataset
