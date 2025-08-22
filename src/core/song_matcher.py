"""
Song Matching Module - Fixed Version
Handles matching input text with song lyrics using multiple strategies
"""

from ngram_processor import NGramProcessor
import re
from collections import defaultdict

class SongMatcher:
    """Handles song matching using multiple search strategies"""
    
    def __init__(self, corpus_manager, similarity_threshold=0.01):
        """
        Initialize song matcher
        Args:
            corpus_manager: CorpusManager instance
            similarity_threshold: Minimum similarity score to consider a match
        """
        self.corpus_manager = corpus_manager
        self.ngram_processor = NGramProcessor()
        self.similarity_threshold = similarity_threshold
        self.preprocessed_songs = {}
        self.keyword_index = defaultdict(list)
        
        # Preprocess all songs
        self.preprocess_corpus()
    
    def preprocess_corpus(self):
        """Preprocess all songs in the corpus"""
        print("Preprocessing song corpus...")
        
        all_songs = self.corpus_manager.get_all_songs()
        print(f"Total songs from corpus manager: {len(all_songs)}")
        
        processed_count = 0
        
        for song_id, song_data in all_songs.items():
            # Extract data from your actual JSON structure - KEEP ORIGINAL KEYS
            title = song_data.get('Title', song_data.get('title', 'Unknown'))
            artist = song_data.get('Artist', song_data.get('artist', 'Unknown'))
            
            # Try multiple keys for lyrics to find the correct one
            lyrics = None
            possible_lyric_keys = ['Lyric', 'lyric', 'Lyrics', 'lyrics', 'text', 'Text']
            
            for key in possible_lyric_keys:
                if key in song_data and song_data[key]:
                    lyrics = song_data[key]
                    break
            
            if not lyrics or not isinstance(lyrics, str) or not lyrics.strip():
                continue
                
            # Store song info - KEEP ORIGINAL KEY NAMES for UI compatibility
            normalized_song_data = {
                'title': title,
                'artist': artist,
                'lyric': lyrics,  # Keep original key name 'lyric' for UI
                'album': song_data.get('Album', song_data.get('album', '')),
                'original_data': song_data
            }
            
            # Extract keywords from title, artist, and lyrics
            all_text = f"{title} {artist} {lyrics}".lower()
            keywords = self._extract_keywords(all_text)
            
            self.preprocessed_songs[song_id] = {
                'song_data': normalized_song_data,
                'keywords': keywords
            }
            
            # Index keywords for fast searching
            for keyword in keywords:
                self.keyword_index[keyword].append(song_id)
            
            processed_count += 1
        
        print(f"Preprocessing complete: {processed_count} songs processed")
        print(f"Keyword index contains {len(self.keyword_index)} unique keywords")
    
    def _extract_keywords(self, text):
        """Extract important keywords from text"""
        # Clean the text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out stop words and short words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'a', 'an', 'is', 'was', 'are', 'were'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def find_matches(self, query_text, max_results=10):
        """
        Find songs that match the input query using multiple strategies
        """
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            print("Empty or invalid query text")
            return []
        
        query_text = query_text.lower().strip()
        print(f"Searching for: '{query_text}'")
        print(f"Available preprocessed songs: {len(self.preprocessed_songs)}")
        
        # Try different search strategies in order of priority
        results = []
        
        # 1. Exact title match (highest priority)
        title_matches = self._search_exact_title(query_text)
        results.extend([(song, 1.0) for song in title_matches])
        
        # 2. Exact artist match
        artist_matches = self._search_exact_artist(query_text)
        results.extend([(song, 0.9) for song in artist_matches])
        
        # 3. Keyword search (works for single words)
        keyword_matches = self._search_by_keywords(query_text)
        results.extend([(song, 0.8) for song in keyword_matches])
        
        # 4. Partial matches in title/artist
        partial_matches = self._search_partial(query_text)
        results.extend([(song, 0.7) for song in partial_matches])
        
        # 5. Lyrics content search
        lyric_matches = self._search_in_lyrics(query_text)
        results.extend([(song, 0.6) for song in lyric_matches])
        
        # Remove duplicates while keeping highest score
        unique_results = {}
        for song_data, score in results:
            song_id = f"{song_data['title']}_{song_data['artist']}"
            if song_id not in unique_results or score > unique_results[song_id][1]:
                unique_results[song_id] = (song_data, score)
        
        # Convert back to list and sort
        final_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(final_results)} matches")
        return final_results[:max_results]
    
    def _search_exact_title(self, query):
        """Search for exact title matches"""
        matches = []
        for song_id, data in self.preprocessed_songs.items():
            if data['song_data']['title'].lower() == query:
                matches.append(data['song_data'])
        return matches
    
    def _search_exact_artist(self, query):
        """Search for exact artist matches"""
        matches = []
        for song_id, data in self.preprocessed_songs.items():
            if data['song_data']['artist'].lower() == query:
                matches.append(data['song_data'])
        return matches
    
    def _search_by_keywords(self, query):
        """Search using keyword index - works for single words"""
        query_words = query.split()
        matched_songs = set()
        
        for word in query_words:
            if word in self.keyword_index:
                for song_id in self.keyword_index[word]:
                    matched_songs.add(song_id)
        
        return [self.preprocessed_songs[song_id]['song_data'] for song_id in matched_songs]
    
    def _search_partial(self, query):
        """Search for partial matches in title and artist"""
        matches = []
        for song_id, data in self.preprocessed_songs.items():
            song_data = data['song_data']
            # Check if query appears in title or artist (case insensitive)
            title_lower = song_data['title'].lower()
            artist_lower = song_data['artist'].lower()
            
            if (query in title_lower or 
                query in artist_lower or
                title_lower in query or
                artist_lower in query):
                matches.append(song_data)
        return matches
    
    def _search_in_lyrics(self, query):
        """Search for query text in lyrics"""
        matches = []
        query_words = query.split()
        
        for song_id, data in self.preprocessed_songs.items():
            lyrics = data['song_data']['lyric'].lower()  # Use 'lyric' key
            
            # Count how many query words appear in lyrics
            match_count = sum(1 for word in query_words if word in lyrics)
            
            # If at least one word matches, include it
            if match_count > 0:
                matches.append(data['song_data'])
        
        return matches
    
    def find_matches_by_artist(self, query_text, artist_name, max_results=5):
        """Find matches within a specific artist's songs"""
        all_matches = self.find_matches(query_text, 50)
        artist_matches = []
        
        for song_data, score in all_matches:
            if artist_name.lower() in song_data['artist'].lower():
                artist_matches.append((song_data, score))
        
        return artist_matches[:max_results]
    
    def find_similar_songs(self, song_id, max_results=5):
        """Find songs similar to a given song"""
        if song_id not in self.preprocessed_songs:
            return []
        
        reference_song = self.preprocessed_songs[song_id]
        reference_data = reference_song['song_data']
        
        # Use title and artist as query
        query = f"{reference_data['title']} {reference_data['artist']}"
        matches = self.find_matches(query, max_results + 1)
        
        # Remove the reference song itself from results
        filtered_matches = [
            (song_data, score) for song_data, score in matches
            if song_data['title'] != reference_data['title'] or song_data['artist'] != reference_data['artist']
        ]
        
        return filtered_matches[:max_results]
    
    def get_match_statistics(self):
        """Get statistics about the matching system"""
        total_songs = len(self.preprocessed_songs)
        return {
            'total_songs': total_songs,
            'keyword_index_size': len(self.keyword_index),
            'similarity_threshold': self.similarity_threshold
        }
    
    def update_similarity_threshold(self, new_threshold):
        """Update similarity threshold for matching"""
        self.similarity_threshold = max(0.0, min(1.0, new_threshold))
    
    def add_song_to_corpus(self, song_data):
        """Add a new song to the corpus and preprocess it"""
        # Extract data from your JSON structure - KEEP ORIGINAL KEYS
        title = song_data.get('Title', song_data.get('title', 'Unknown'))
        artist = song_data.get('Artist', song_data.get('artist', 'Unknown'))
        
        # Try multiple keys for lyrics
        lyrics = None
        possible_lyric_keys = ['Lyric', 'lyric', 'Lyrics', 'lyrics', 'text', 'Text']
        
        for key in possible_lyric_keys:
            if key in song_data and song_data[key]:
                lyrics = song_data[key]
                break
        
        if not lyrics or not isinstance(lyrics, str) or not lyrics.strip():
            return None
        
        # Generate unique song ID
        song_id = f"{artist}_{title}".lower().replace(' ', '_').replace("'", "")
        
        # Store song data - KEEP ORIGINAL KEY NAMES
        normalized_song_data = {
            'title': title,
            'artist': artist,
            'lyric': lyrics,  # Keep original key name
            'album': song_data.get('Album', song_data.get('album', '')),
            'original_data': song_data
        }
        
        # Extract keywords
        all_text = f"{title} {artist} {lyrics}".lower()
        keywords = self._extract_keywords(all_text)
        
        self.preprocessed_songs[song_id] = {
            'song_data': normalized_song_data,
            'keywords': keywords
        }
        
        # Update keyword index
        for keyword in keywords:
            self.keyword_index[keyword].append(song_id)
        
        return song_id