import os
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import muspy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_genres(spotify, artist_name):
    results = spotify.search(q='artist:' + artist_name, limit=1, type='artist')
    print(results)
    if len(results['artists']['items']) > 0:
        return results['artists']['items'][0]['genres']
    else:
        return []

def load_match_dataframes():
    lakh_msd_match = pd.read_csv('matched_ids.txt', sep=';',
        header=None, names=['lakh_track_id', 'msd_track_id'])
    msd_tracks = pd.read_csv('unique_tracks.txt',   sep='<SEP>', engine='python',
        header=None, names=['track_id','song_id','artist_name','song_name'])
    return lakh_msd_match, msd_tracks

def get_converted_dataset():
    dataset_root_path = Path(os.environ['DATASET_ROOT_PATH'])
    must_download_and_extract = not os.path.exists(os.path.join(dataset_root_path, 'lmd_matched'))
    lakh_matched_dataset = muspy.LakhMIDIMatchedDataset(
        dataset_root_path, download_and_extract=must_download_and_extract,
        cleanup=True
    )
    if not lakh_matched_dataset.converted_exists():
        lakh_matched_dataset.convert(n_jobs=-1, verbose=True)
    return lakh_matched_dataset

def get_track_artist_title_and_ids(music_track, lakh_msd_match, msd_tracks):
    # Obtain track id
    track_id = music_track.metadata.source_filename[:-4] # remove '.mid'
    # Look for track id in the match dataset. Note that the same track id
    # may be linked to multiple versions of the same song (or remixes, or even
    # other similar songs). We might do a confidence-based selection, but since
    # we don't have a confidence measure we just stick to picking the first
    # available match.
    msd_track_ids = lakh_msd_match[lakh_msd_match['lakh_track_id'] == track_id]['msd_track_id']
    if len(msd_track_ids) > 0:
        msd_track_id = msd_track_ids.iloc[0]
    else:
        return None, None, None, None
    # Obtain artist name from the MSD metadata dataset
    names = msd_tracks[msd_tracks['track_id'] == msd_track_id][['artist_name', 'song_name']]
    return names['artist_name'].values[0], names['song_name'].values[0], track_id, msd_track_id

if __name__ == '__main__':
    track_genre_match = {}
    # Load the .env file where credentials are tracked.
    # We need to have both a SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET
    load_dotenv()
    # Create a Spotify object to access the API
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    # Optionally download, extract and convert the dataset
    lakh_matched_dataset = get_converted_dataset()
    # Load the file linking Lakh tracks to the One Million Songs Dataset
    # and the OMS Dataset's list of unique tracks to find the appropriate
    # artist names for each track
    lakh_msd_match, msd_tracks = load_match_dataframes()
    # Iterate over the music in the dataset
    for music_track in tqdm(lakh_matched_dataset):
        # Try to obtain artist name
        artist_name, song_name, track_id, msd_track_id = \
            get_track_artist_title_and_ids(music_track, lakh_msd_match, msd_tracks)
        if artist_name is not None:
            # Search the artist name on Spotify to get the genres they're associated with.
            artist_genres = get_genres(spotify, artist_name)
            print(f"\nFound genres {artist_genres} for track {song_name}")
            track_genre_match[track_id] = artist_genres
    # Save the matches track_id -> genres
    with open('ids_to_genres.json', 'w') as f:
        json.dump(track_genre_match, f)
