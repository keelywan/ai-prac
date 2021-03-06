import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df_data = pd.read_csv("./mood_data.csv", sep=',', header=None,
                      encoding='unicode_escape', thousands=',')
df_data.columns = df_data.iloc[0]
df_data = df_data.drop(0)


def powerset(s):
    '''
    Takes in a list of features and returns a list of all combinations of subsets of features, 
    excluding the empty set.
    '''
    x = len(s)
    out = []
    for i in range(1, 1 << x):
        out.append([s[j] for j in range(x) if (i & (1 << j))])
    return out


# selecting which variables to include - removed time_signature, mode, key
variables = ["acousticness",
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
             "start_of_fade_out"]

X = df_data[variables]
Y = df_data['mood']

powerset_groups = powerset(variables)

qt = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit(X)
X = qt.transform(X)

xTr, xTe, yTr, yTe = train_test_split(X, Y, test_size=0.3)


def transform_song(s):
    new_song = s[variables]
    return qt.transform(new_song)


def get_data_opt_var():
    return xTr, xTe, yTr, yTe


def powerset_get_data():
    return map(lambda x: (x, df_data[x], Y), powerset_groups)
