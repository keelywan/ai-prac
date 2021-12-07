import sys
import pandas as pd
from joblib import load

# Gets the audio features via command line arguments 
data = [[float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
        float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]), 
        float(sys.argv[11])]]
df = pd.DataFrame(data, columns=['acousticness', 'danceability', 'energy',
        'instrumentalness', 'key', 'loudness', 'mode', 'speechiness', 'tempo', 
        'time_signature', 'valence'])

# clf = load('./KNN-0.604-70-auto-distance.joblib')
# print(clf.predict(df[0:1]))
print("SUCCESS")

