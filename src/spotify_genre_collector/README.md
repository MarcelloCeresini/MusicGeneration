# Genre Collector

We use Spotify's API to collect the most appropriate genres given the composer(s) of the songs in the Lakh MIDI dataset. 

Spotify assign a genre to artists, not to single tracks: therefore we need to find the artist of a given song and look it up using the API provided through the `spotipy` library.

First of all, we need credentials to use the API: in this folder there should be a `.env` file containing the following environment variables:

```
SPOTIPY_CLIENT_ID=...
SPOTIPY_CLIENT_SECRET=...
DATASET_ROOT_PATH=...
```

The client ID and Secret are obtained after creating an app on [Spotify's developer dashboard](https://developer.spotify.com/dashboard/applications).

We use the [Lakh MIDI Dataset Matched](https://colinraffel.com/projects/lmd/), which contains 115,190 MIDI files that have been matched with the [Million Song Dataset](http://millionsongdataset.com/). Since Lakh is not consistent with its annotations regarding the artists of the songs, we try to resolve the artist name by looking at the song's metadata in the Million Song Dataset. 
In particular, we use two `csv` datasets:
- `matched_ids.txt`, whose original version is available [here](https://drive.google.com/uc?id=1yTeqvZ1HM1PGVm_jHPU3Rxb8lh3ctzn8), contains the list of all matches that have been found between the Lakh dataset and the Million Song Dataset in terms of track IDs. 
In our version, we simply changed the separator (from 4 spaces to `;`). 
We use it to translate Lakh track IDs into MSD track IDs. Unfortunately, the match is not 1-to-1: since the matches were collected algorithmically, all matches over a certain threshold are maintained. Since we don't have a confidence measure and re-doing the match would be too expensive, we simply take the first available match as the correct one.
- `unique_tracks.txt`, available [here](http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt), contains track IDs, track names and artist names for each of the songs in the MSD. We use it to find the artist name given the track ID. We did not modify this file.

Through these files we try to obtain the artist names so that we can query the Spotify database using their API to obtain a likely genre for the track. Finally, we save the track-genre matches for future analysis.