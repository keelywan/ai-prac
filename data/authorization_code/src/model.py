import sys
import data_process
import pandas as pd
from joblib import load

# Gets the audio features via command line arguments 
data = [[float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
        float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]), 
        float(sys.argv[11])]]
df = pd.DataFrame(data, columns=['acousticness', 'danceability', 'energy',
        'instrumentalness', 'key', 'loudness', 'mode', 'speechiness', 'tempo', 
        'time_signature', 'valence'])
song = data_process.transform_song(df[0:1])
clf = load('./KNN-0.612-70-ball_tree-uniform.joblib')
print(clf.predict(song))

