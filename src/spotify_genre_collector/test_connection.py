from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

if __name__ == '__main__':
    # We need to have both a SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET
    load_dotenv()
    # Create a Spotify object to access the API
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    results = spotify.search(q='artist:' + 'radiohead', limit=1, type='artist')
    print(results)
