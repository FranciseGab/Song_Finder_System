"""
Corpus Manager Module
Handles loading and managing song corpus data
"""

import json
import os
import pickle
from collections import defaultdict

class CorpusManager:
    """Manages song corpus data and provides access methods"""

    def __init__(self):
        self.songs = {}  # All songs with unique IDs
        self.artists = defaultdict(list)  # artist -> list of song IDs
        self.albums = defaultdict(list)   # album -> list of song IDs
        self.corpus_loaded = False

    def load_corpus(self, corpus_directory):
        """Load all JSON files from the corpus directory"""
        if not os.path.exists(corpus_directory):
            raise FileNotFoundError(f"Corpus directory not found: {corpus_directory}")

        print(f"Loading corpus from: {corpus_directory}")
        loaded_files = 0
        total_songs = 0

        json_files = [f for f in os.listdir(corpus_directory) if f.endswith('.json')]
        for filename in json_files:
            file_path = os.path.join(corpus_directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Default artist name from filename
                artist_name = os.path.splitext(filename)[0].replace('_', ' ').title()

                songs_loaded = self._process_json_data(data, artist_name, filename)
                total_songs += songs_loaded
                loaded_files += 1

                print(f"  Loaded {songs_loaded} songs from {filename}")

            except Exception as e:
                print(f"  Error loading {filename}: {str(e)}")
                continue

        self.corpus_loaded = True
        print(f"Corpus loading complete: {loaded_files} files, {total_songs} total songs")
        print(f"Artists in corpus: {len(self.artists)}")

    def _process_json_data(self, data, default_artist, filename):
        """Process JSON data and extract songs"""
        songs_processed = 0

        # Determine list of songs
        if isinstance(data, dict):
            if 'songs' in data:
                songs_list = data['songs']
            elif 'data' in data:
                songs_list = data['data']
            elif any(k.lower() in data for k in ['title', 'lyric', 'artist']):
                songs_list = [data]
            else:
                songs_list = list(data.values())
                if not isinstance(songs_list[0], dict):
                    print(f"  Warning: Unexpected JSON structure in {filename}")
                    return 0
        elif isinstance(data, list):
            songs_list = data
        else:
            print(f"  Warning: Unexpected data type in {filename}")
            return 0

        for song_data in songs_list:
            if self._add_song(song_data, default_artist, filename):
                songs_processed += 1

        return songs_processed

    def _add_song(self, song_data, default_artist, source_file):
        """Add a single song to the corpus"""
        if not isinstance(song_data, dict):
            return False

        # Extract song info with fallbacks
        title = song_data.get('Title') or song_data.get('title') or song_data.get('name') or 'Unknown Title'
        artist = song_data.get('Artist') or song_data.get('artist') or song_data.get('Singer') or default_artist or 'Unknown Artist'
        lyrics = song_data.get('Lyric') or song_data.get('lyric') or song_data.get('text') or song_data.get('content') or ''

        if not lyrics.strip():
            return False  # skip songs without lyrics

        # Unique song ID
        song_id = f"{artist}_{title}".lower().replace(' ', '_').replace("'", "")

        # Standardized song object (all lowercase keys)
        standardized_song = {
            'id': song_id,
            'title': title,
            'artist': artist,
            'lyrics': lyrics,
            'album': song_data.get('Album') or song_data.get('album') or '',
            'year': song_data.get('Year') or song_data.get('year') or '',
            'genre': song_data.get('Genre') or song_data.get('genre') or '',
            'duration': song_data.get('Duration') or song_data.get('duration') or '',
            'source_file': source_file
        }

        self.songs[song_id] = standardized_song
        self.artists[artist.lower()].append(song_id)
        album = standardized_song['album']
        if album:
            self.albums[album.lower()].append(song_id)

        return True

    # ---------------- Access Methods ----------------

    def get_all_songs(self):
        return self.songs

    def get_song_by_id(self, song_id):
        return self.songs.get(song_id)

    def get_songs_by_artist(self, artist_name):
        artist_key = artist_name.lower()
        return {sid: self.songs[sid] for sid in self.artists.get(artist_key, [])}

    def get_songs_by_album(self, album_name):
        album_key = album_name.lower()
        return {sid: self.songs[sid] for sid in self.albums.get(album_key, [])}

    def search_songs_by_title(self, title_query):
        query = title_query.lower()
        return {sid: s for sid, s in self.songs.items() if query in s['title'].lower()}

    def get_artist_list(self):
        return list(self.artists.keys())

    def get_album_list(self):
        return list(self.albums.keys())

    # ---------------- Statistics & Cache ----------------

    def get_corpus_statistics(self):
        total_songs = len(self.songs)
        total_artists = len(self.artists)
        total_albums = len([album for album in self.albums.keys() if album])

        total_lyrics_length = sum(len(song['lyrics']) for song in self.songs.values())
        songs_with_albums = sum(1 for song in self.songs.values() if song['album'])
        avg_lyrics_length = total_lyrics_length / total_songs if total_songs else 0

        return {
            'total_songs': total_songs,
            'total_artists': total_artists,
            'total_albums': total_albums,
            'songs_with_albums': songs_with_albums,
            'avg_lyrics_length': avg_lyrics_length,
            'corpus_loaded': self.corpus_loaded
        }

    def save_corpus_cache(self, cache_file_path):
        try:
            cache_data = {
                'songs': self.songs,
                'artists': dict(self.artists),
                'albums': dict(self.albums),
                'corpus_loaded': self.corpus_loaded
            }
            with open(cache_file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Corpus cache saved to: {cache_file_path}")
            return True
        except Exception as e:
            print(f"Error saving corpus cache: {str(e)}")
            return False

    def load_corpus_cache(self, cache_file_path):
        try:
            if not os.path.exists(cache_file_path):
                return False
            with open(cache_file_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.songs = cache_data.get('songs', {})
            self.artists = defaultdict(list, cache_data.get('artists', {}))
            self.albums = defaultdict(list, cache_data.get('albums', {}))
            self.corpus_loaded = cache_data.get('corpus_loaded', False)
            print(f"Corpus cache loaded from: {cache_file_path}")
            print(f"Loaded {len(self.songs)} songs from cache")
            return True
        except Exception as e:
            print(f"Error loading corpus cache: {str(e)}")
            return False

    def export_corpus_summary(self, output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Song Corpus Summary\n")
                f.write("==================\n\n")
                stats = self.get_corpus_statistics()
                f.write(f"Total Songs: {stats['total_songs']}\n")
                f.write(f"Total Artists: {stats['total_artists']}\n")
                f.write(f"Total Albums: {stats['total_albums']}\n\n")
                f.write("Artists and Song Counts:\n")
                f.write("-" * 30 + "\n")
                for artist, song_ids in sorted(self.artists.items()):
                    f.write(f"{artist.title()}: {len(song_ids)} songs\n")
                f.write(f"\nCorpus summary exported to: {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting corpus summary: {str(e)}")
            return False
