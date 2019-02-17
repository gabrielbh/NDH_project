# import spotipy as sp
# from spotipy.oauth2 import SpotifyClientCredentials
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.ensemble import ExtraTreesClassifier

PASS_SCORE = 22559.3

# initilize spotify connection:
# client_credentials_manager = SpotifyClientCredentials(client_id='87158cc41aa945e0b7be77d94a57efa8', client_secret='fd77db134a3a45b7b3f1fa85f43375b9')
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# def get_spotify_features(track_name, artist_name, counter):
#     # conter: number of unfound songs
#     track = sp.search(q='artist:' + artist_name + ' track:' + track_name, type='track', limit=1)
#
#     if len(track['tracks']['items']) == 0:
#         counter += 1
#         return counter, None
#
#     id = track['tracks']['items'][0]['id']
#     features = sp.audio_features(id)
#     return counter, features


# saves dictionary object to memory
# obj - object to save
# name - file name
def save_obj(obj, name):
    f = open('obj/'+ name + '.pkl', 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# loads dictionary object from memory
# obj - object to save
# name - file name
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# creates dict from artist name + song name lists from billboard
def create_dict_features_billboard(name_of_list_in_memory):
    dict_songs = {}
    file = open(name_of_list_in_memory, 'r')
    counter = 0
    for line in file:
        line = line.split(", ")
        song_name = line[0]
        artist_name = line[1].replace('\n', "")
        counter, features = get_spotify_features(song_name, artist_name, counter)
        if features is not None:
            dict_songs[song_name, artist_name] = features
    save_obj(dict_songs, 'billboard_song_features_dict')



def create_dict_features_arbitrary(name_of_list_in_memory):
    dict_songs = {}
    file = open(name_of_list_in_memory, 'r')
    unfound_songs = 0
    all_songs = 0
    for line in file:
        line = line.split(",")
        if len(line) == 24:
            song_name = line[-3].lower()
            artist_name = line[7].lower()
        else:
            # if some data in the set is lost:
            song_name = line[-3].lower()
            artist_name = line[6].lower()
        song_name = re.sub("[\(\[].*?[\)\]]", "", song_name)
        artist_name = re.sub("[\(\[].*?[\)\]]", "", artist_name)
        unfound_songs, features = get_spotify_features(song_name, artist_name, unfound_songs)
        if features is not None:
            dict_songs[song_name, artist_name] = features
        else:
            unfound_songs += 1
        print("all song: ", all_songs)
        print("failed: ", unfound_songs)
        all_songs += 1
    save_obj(dict_songs, 'arbitrary_songs_features_dict')

def linearRegression(M, Y):
    """
    Returns the prediction vector.
    M is a matrix of samples, where each line is a sample and his differente features,
    Y is the vector which indicates if the sample is healthy or not.
    """

    return np.dot(np.linalg.pinv(M), Y)


def calc_weights():
    features_name = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                     'liveness', 'valence', 'tempo', 'duration_ms']

    dict_all_songs = load_obj('arbitrary_songs_features_dict')
    dict_billboard = load_obj('bilboard_dict')

    data = []
    tags = []
    sum = 0

    for song in dict_all_songs:
        if song in dict_billboard.keys():
            tag = 1 #hit
            sum += 1
        else:
            tag = 0 #not hit
        features = []
        for feature in features_name:
            if feature == 'duration_ms':
                features.append(dict_all_songs[song][0][feature] / 60000)
                dict_all_songs[song][0][feature] /= 60000
            else:
                features.append(dict_all_songs[song][0][feature])
        data.append(features)
        tags.append(tag)
    for song in dict_billboard:
        tag = 1
        features = []
        for feature in features_name:
            if feature == 'duration_ms':
                features.append(dict_billboard[song][0][feature] / 60000)
                dict_billboard[song][0][feature] /= 60000
            else:
                features.append(dict_billboard[song][0][feature])
        data.append(features)
        tags.append(tag)
    data = np.array(data)
    tags = np.array(tags)

    rl = linearRegression(data, tags)
    model = ExtraTreesClassifier()
    model.fit(data, tags)
    feature_weights = model.feature_importances_
    feature_weights = rl
    feature_dict = {}
    for i in range(len(features_name)):
        feature_dict[features_name[i]] = feature_weights[i]
    # feature_dict = {key: weight for key in features_name for weight in feature_weights}
    return feature_dict, dict_billboard, dict_all_songs


def compute_single_song_score(song, features, weights, dict_features, opt_weights):
    score = 0
    for i in range(len(features)):
        score += weights[i] * abs(dict_features[song][0][features[i]] - opt_weights[features[i]])
    return score


def compute_score(song_features_dict, feature_weights):
    dict_features = song_features_dict
    features_name = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                     'liveness', 'valence', 'tempo', 'duration_ms']
    mean = find_hits_mean(dict_billboard, list(w_dict.keys()))
    scores = []
    for song in dict_features.keys():
        # score = 0
        for i in range(len(features_name)):
            score = compute_single_song_score(song, features_name, feature_weights, dict_features, mean)
            # score += feature_weights[i] * dict_features[song][0][features_name[i]]
        scores.append(score)
    return scores


def find_hits_mean(hits, features):
    """
    hits of specific genre (pop, hip hop, rap)
    :return: dict with the mean value of hits for each feature.
    """
    dict_hits_features_mean = {f : 0 for f in features}
    dict_hits_score = {f : 0 for f in features}
    len_of_hits = len(hits)
    for song in hits:
        for feature in features:
            dict_hits_features_mean[feature] += hits[song][0][feature]
    for feature in features:
        dict_hits_features_mean[feature] /= len_of_hits
    return dict_hits_features_mean



def greedy(song, weight_dict, dict_features, hits_mean_dict, rng=(0.04415809256636373, 0.06413266371976678)):
    changes = {}
    # s_weights = sorted(weight_dict.values())
    s_weights = sorted(weight_dict.items(), key=lambda x:x[1], reverse=True)
    weights = [weight[1] for weight in s_weights]
    features = [feature[0] for feature in s_weights]
    # if compute_single_song_score(song, features, weights, dict_features) - PASS_SCORE >= delta:
    #     return changes
    done = False
    for i in range(len(weights)):
        original_value = dict_features[song][0][features[i]]
        current_value = original_value
        hit_feature_mean = hits_mean_dict[features[i]]
        dict_features[song][0][features[i]] = hit_feature_mean
        score = compute_single_song_score(song, features, weights, dict_features, hits_mean_dict)
        dict_features[song][0][features[i]] = hit_feature_mean

        changes[features[i]] = hit_feature_mean - original_value
        if score in range(rng):
            return changes
    return "You're a failure you cunt"

w_dict, dict_billboard, dict_all_songs = calc_weights()
mean = find_hits_mean(dict_billboard, list(w_dict.keys()))
print(greedy(list(dict_billboard.keys())[0], w_dict, dict_billboard,mean))
hits = (compute_score(dict_billboard, list(w_dict.values())))

mnH = np.mean(hits)
print("mean Hits: " + str(mnH))
print(max(hits))
print(min(hits))

songs = compute_score(dict_all_songs, list(w_dict.values()))
mnS = np.mean(songs)
print("mean songs: " + str(mnS))
print("max: " + str(max(songs)))
print(min(songs))

std = np.std(hits)

print("hits std: " + str(std))
stdS = np.std(songs)
print("songs std: " + str(stdS))

print(len(songs))
print(len([x for x in songs if x < mnH + 0.25*std and x > mnH - 0.25*std]))
print("range = " + " min : " + str(mnH - 0.25*std) + " max : " + str(mnH + 0.25*std))



# # TODO:
# # 1. creating data of hit songs and non-hit songs (with labels)
# # 2. finding feature weights with des trees.
# # 3. score function: songs score = sum(feature*weight) for each feature.
# # 4. ploting scores of all songs (or hit and non hits separately) to see if a hit sing has different score.
# # 5. algorithm: finding the min path for reaching the hit-song-score-range for a new song
#
#
#
#
# def analyze_data(song_features_dict):
#     dict_features = load_obj(song_features_dict)
#     danceability = []
#     energy = []
#     key = [0] * 12
#     loudness = []
#     mode = [0] * 2
#     speechiness = []
#     acousticness = []
#     # instrumentalness = []
#     liveness = []
#     valence = []
#     tempo = []
#     # type = []
#     duration_ms = []
#
#     for dict_key in dict_features.keys():
#         danceability.append(dict_features[dict_key][0]['danceability'])
#         energy.append(dict_features[dict_key][0]['energy'])
#         key[dict_features[dict_key][0]['key']] += 1
#         loudness.append(dict_features[dict_key][0]['loudness'])
#         mode[dict_features[dict_key][0]['mode']] += 1
#         speechiness.append(dict_features[dict_key][0]['speechiness'])
#         acousticness.append(dict_features[dict_key][0]['acousticness'])
#         # instrumentalness.append(dict_features[dict_key][0]['instrumentalness'])
#         liveness.append(dict_features[dict_key][0]['liveness'])
#         valence.append(dict_features[dict_key][0]['valence'])
#         tempo.append(dict_features[dict_key][0]['tempo'])
#         # type.append(dict_features[dict_key][0]['type'])
#         duration_ms.append(dict_features[dict_key][0]['duration_ms'])
#
#
#     num_samples = len(dict_features)
#     create_graph(danceability, 'danceability_graph')
#     create_graph(energy, 'energy_graph')
#     create_key_graph(np.array(key) / num_samples, 'key_graph')
#     create_graph(np.array(loudness) / 60.0, 'loudness_graph')
#     create_mode_graph(np.array(mode) / num_samples, 'mode_graph')
#     create_graph(speechiness, 'speechiness_graph')
#     create_graph(acousticness, 'acousticness_graph')
#     # create_graph(instrumentalness, 'instrumentalness_graph')
#     create_graph(liveness, 'liveness_graph')
#     create_graph(valence, 'valence_graph')
#     create_graph(np.array(tempo) / max(tempo), 'tempo_graph')
#     # create_graph(type, 'type_graph')
#     create_graph(np.array(duration_ms) / max(duration_ms), 'duration_ms_graph')
#     # print("key: ", np.array(key) / num_samples)
#     # print("mode: ", np.array(mode) / num_samples)
#
#     features_name = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
#                      'liveness', 'valence', 'tempo', 'duration_ms']
#
#
# # Danceability: describes how suitable a track is for dancing based on a combination of musical elements including
# # tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most
# # danceable.
# def create_graph(data_list, fig_name):
#     mean = round(np.mean(data_list), 4)
#     plt.figure()
#     plt.title(fig_name + ", mean = " + str(mean))
#     plt.hist(data_list)
#     plt.savefig(fig_name)
#
#
# def create_key_graph(keys_list, fig_name):
#     plt.figure()
#     plt.title('key_graph')
#     # colors=('deeppink', 'red', 'darkorange', 'gold', 'yellow', 'greenyellow', 'limegreen', 'mediumspringgreen', 'turquoise', 'deepskyblue', 'royalblue', 'darkorchid','deeppink')
#     color=cm.rainbow(np.linspace(0, 1, 12))
#     plt.bar(['C','C♯,D♭','D','D♯,E♭','E','F','F♯,G♭','G','G♯,A♭','A','A♯,B♭', 'B'], keys_list, color=color)
#     plt.savefig(fig_name)
#
#
# def create_mode_graph(mode_list, fig_name):
#     plt.figure()
#     plt.title('mode_graph')
#     plt.bar(['minor', 'major'], mode_list, color=('aquamarine', 'mediumpurple'))
#     plt.savefig(fig_name)
#
#