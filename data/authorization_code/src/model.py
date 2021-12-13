import sys
import data_process
import pandas as pd
from joblib import load

# Gets the audio features via command line arguments
data = [[float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
        float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]),
        float(sys.argv[11]), float(sys.argv[12]), float(sys.argv[13]), float(sys.argv[14]), float(sys.argv[15]),
        float(sys.argv[16]), float(sys.argv[17])]]
df = pd.DataFrame(data, columns=["acousticness",
                                 "danceability",
                                 "duration_ms",
                                 "energy",
                                 "instrumentalness",
                                 "key",
                                 "liveness",
                                 "loudness",
                                 "speechiness",
                                 "tempo",
                                 "valence",
                                 "key_confidence",
                                 "mode_confidence",
                                 "tempo_confidence",
                                 "time_signature_confidence",
                                 "end_of_fade_in",
                                 "start_of_fade_out"])
song = data_process.transform_song(df[0:1])
clf = load("KNN-0.652-40-auto-uniform.joblib")
print(clf.predict(song))
