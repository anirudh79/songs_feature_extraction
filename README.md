# songs_feature_extraction

For feature extraction I used the python library Librosa.

I used different library methods to get 24 features over a 30 second (~1290 frames) interval.

Taking all 1290 frames for each song wouldnt have been feasible as it would lead to 129000 training examples for only a 100 songs.
Moreover,no decipherable pattern would be formed over a single frame. Taking the mean and max over the 30 second interval would make it hard to distinguish between differnt genres as the interval would be too big.

Therefore, I used 1 second intervals which leads to 30 examples per song. I used both mean and max of the signals to account for the large spikes in signal above the mean,if any.

For each song the 48 features so obtained and the training label is added to a dataset.

The classifier would predict the genre of the song as the genre which is classified the most number of times over the 30 exapmles.
Example: A classical song can be predicted in 20 of its 1 second intervals as classical, 5 times as rock and 5 times as pop leading to an overall prediction as classical with a 66.66% confidence for that particular song.
