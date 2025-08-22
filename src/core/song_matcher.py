"""
Song Matching Module - Debug Version
Handles matching input text with song lyrics using n-grams
"""

# Import n-gram processor
from ngram_processor import NGramProcessor

class SongMatcher:
    """Handles song matching using n-gram similarity"""
    
    def __init__(self, corpus_manager, similarity_threshold=0.05):  # Lower threshold for testing
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
        
        # Preprocess all songs for faster matching
        self.preprocess_corpus()
    
    def preprocess_corpus(self):
        """Preprocess all songs in the corpus to generate n-grams"""
        print("Preprocessing song corpus...")
        
        all_songs = self.corpus_manager.get_all_songs()
        print(f"Total songs from corpus manager: {len(all_songs)}")
        
        processed_count = 0
        empty_lyrics_count = 0
        failed_ngram_count = 0
        
        # Debug: Show a few sample songs
        sample_count = 0
        for song_id, song_data in all_songs.items():
            sample_count += 1
            if sample_count <= 3:  # Show first 3 songs for debugging
                print(f"Sample song {sample_count}:")
                print(f"  ID: {song_id}")
                print(f"  Keys: {list(song_data.keys())}")
                if 'Lyric' in song_data:
                    lyric_preview = song_data['Lyric'][:100] if song_data['Lyric'] else "EMPTY"
                    print(f"  Lyric preview: {lyric_preview}...")
                print()
            
            # Try multiple possible keys for lyrics
            lyrics = None
            possible_lyric_keys = ['Lyric', 'lyric', 'Lyrics', 'lyrics', 'text', 'Text']
            
            for key in possible_lyric_keys:
                if key in song_data and song_data[key]:
                    lyrics = song_data[key]
                    break
            
            if not lyrics or not lyrics.strip():
                empty_lyrics_count += 1
                continue
                
            # Generate n-grams for the song
            try:
                ngrams_dict = self.ngram_processor.process_text_to_ngrams(lyrics)
                if not ngrams_dict:  # Empty n-grams dict
                    failed_ngram_count += 1
                    continue
                    
                ngram_counters = self.ngram_processor.ngrams_to_counter(ngrams_dict)
                
                # Normalize the song data keys for consistency
                normalized_song_data = {
                    'title': song_data.get('Title', song_data.get('title', 'Unknown')),
                    'artist': song_data.get('Artist', song_data.get('artist', 'Unknown')),
                    'album': song_data.get('Album', song_data.get('album', '')),
                    'lyric': lyrics,
                    'original_data': song_data
                }
                
                self.preprocessed_songs[song_id] = {
                    'song_data': normalized_song_data,
                    'ngrams': ngrams_dict,
                    'ngram_counters': ngram_counters,
                    'keywords': self.ngram_processor.extract_keywords(lyrics)
                }
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing song {song_id}: {str(e)}")
                failed_ngram_count += 1
                continue
        
        print(f"Preprocessing complete:")
        print(f"  Successfully processed: {processed_count} songs")
        print(f"  Empty lyrics: {empty_lyrics_count} songs")
        print(f"  Failed n-gram generation: {failed_ngram_count} songs")
        print(f"  Total attempted: {len(all_songs)} songs")
        
        # Show some preprocessed song info for debugging
        if processed_count > 0:
            sample_processed = list(self.preprocessed_songs.items())[:2]
            print("\nSample preprocessed songs:")
            for song_id, data in sample_processed:
                song_info = data['song_data']
                print(f"  {song_info['title']} by {song_info['artist']}")
                print(f"    N-gram sizes: {list(data['ngrams'].keys())}")
                print(f"    Keywords: {data['keywords'][:5]}...")  # First 5 keywords
    
    def find_matches(self, query_text, max_results=311):
        """
        Find songs that match the input query
        Args:
            query_text: User input text/lyrics
            max_results: Maximum number of results to return
        Returns:
            List of (song_data, similarity_score) tuples sorted by score
        """
        print(f"\nSearching for: '{query_text}'")
        print(f"Available preprocessed songs: {len(self.preprocessed_songs)}")
        
        if not query_text.strip():
            return []
        
        # Process query text
        query_ngrams = self.ngram_processor.process_text_to_ngrams(query_text)
        if not query_ngrams:
            print("Failed to generate n-grams for query")
            return []
        
        print(f"Query n-gram sizes: {list(query_ngrams.keys())}")
        query_counters = self.ngram_processor.ngrams_to_counter(query_ngrams)
        query_keywords = self.ngram_processor.extract_keywords(query_text)
        print(f"Query keywords: {query_keywords}")
        
        # Calculate similarity with each song
        matches = []
        scores_calculated = 0
        
        for song_id, preprocessed_data in self.preprocessed_songs.items():
            song_data = preprocessed_data['song_data']
            song_counters = preprocessed_data['ngram_counters']
            song_keywords = preprocessed_data['keywords']
            
            # Calculate n-gram similarity
            ngram_scores = {}
            for n in self.ngram_processor.n_sizes:
                if n in query_counters and n in song_counters:
                    # Use both Jaccard and Cosine similarity
                    jaccard_score = self.ngram_processor.calculate_ngram_similarity(
                        query_counters[n], song_counters[n], 'jaccard'
                    )
                    cosine_score = self.ngram_processor.calculate_ngram_similarity(
                        query_counters[n], song_counters[n], 'cosine'
                    )
                    
                    # Combine both similarity measures
                    combined_score = (jaccard_score * 0.6) + (cosine_score * 0.4)
                    ngram_scores[n] = combined_score
            
            if ngram_scores:
                scores_calculated += 1
                
                # Calculate overall n-gram similarity
                ngram_similarity = self.ngram_processor.combine_ngram_scores(ngram_scores)
                
                # Calculate keyword similarity
                keyword_similarity = self._calculate_keyword_similarity(query_keywords, song_keywords)
                
                # Calculate phrase similarity (exact phrase matching)
                phrase_similarity = self._calculate_phrase_similarity(query_text, song_data.get('lyric', ''))
                
                # Combine all similarity measures with weights
                final_score = (
                    ngram_similarity * 0.6 +      # N-gram similarity (main component)
                    keyword_similarity * 0.25 +    # Keyword matching
                    phrase_similarity * 0.15       # Exact phrase matching
                )
                
                # Debug: Show high-scoring matches
                if final_score > 0.01:  # Very low threshold for debugging
                    print(f"  Potential match: {song_data['title']} by {song_data['artist']}")
                    print(f"    N-gram: {ngram_similarity:.3f}, Keyword: {keyword_similarity:.3f}, Phrase: {phrase_similarity:.3f}")
                    print(f"    Final score: {final_score:.3f}")
                
                # Only include matches above threshold
                if final_score >= self.similarity_threshold:
                    matches.append((song_data, final_score))
        
        print(f"Calculated scores for {scores_calculated} songs")
        print(f"Found {len(matches)} matches above threshold {self.similarity_threshold}")
        
        # Sort by similarity score (descending) and limit results
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_results]
    
    def _calculate_keyword_similarity(self, query_keywords, song_keywords):
        """Calculate similarity based on keyword matching"""
        if not query_keywords or not song_keywords:
            return 0.0
        
        query_set = set(query_keywords)
        song_set = set(song_keywords)
        
        # Jaccard similarity for keywords
        intersection = len(query_set.intersection(song_set))
        union = len(query_set.union(song_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_phrase_similarity(self, query_text, song_lyrics):
        """Calculate similarity based on exact phrase matching"""
        if not query_text or not song_lyrics:
            return 0.0
        
        # Clean both texts
        clean_query = self.ngram_processor.preprocess_text(query_text)
        clean_lyrics = self.ngram_processor.preprocess_text(song_lyrics)
        
        if not clean_query or not clean_lyrics:
            return 0.0
        
        # Check for exact phrase matches
        query_words = clean_query.split()
        
        # Look for exact sequences in lyrics
        max_match_length = 0
        for i in range(len(query_words)):
            for j in range(i + 1, len(query_words) + 1):
                phrase = ' '.join(query_words[i:j])
                if phrase in clean_lyrics:
                    max_match_length = max(max_match_length, len(phrase.split()))
        
        # Normalize by query length
        return max_match_length / len(query_words) if query_words else 0.0
    
    # ... (keeping the rest of the methods the same)
    def find_matches_by_artist(self, query_text, artist_name, max_results=5):
        """Find matches within a specific artist's songs"""
        artist_songs = self.corpus_manager.get_songs_by_artist(artist_name)
        if not artist_songs:
            return []
        
        # Filter preprocessed songs to only include this artist
        filtered_songs = {
            song_id: data for song_id, data in self.preprocessed_songs.items()
            if data['song_data'].get('artist', '').lower() == artist_name.lower()
        }
        
        # Temporarily replace preprocessed songs for matching
        original_songs = self.preprocessed_songs
        self.preprocessed_songs = filtered_songs
        
        try:
            matches = self.find_matches(query_text, max_results)
            return matches
        finally:
            # Restore original preprocessed songs
            self.preprocessed_songs = original_songs
    
    def find_similar_songs(self, song_id, max_results=5):
        """Find songs similar to a given song"""
        if song_id not in self.preprocessed_songs:
            return []
        
        reference_song = self.preprocessed_songs[song_id]
        reference_lyrics = reference_song['song_data'].get('lyric', '')
        
        # Use the song's lyrics as query
        matches = self.find_matches(reference_lyrics, max_results + 1)
        
        # Remove the reference song itself from results
        filtered_matches = [
            (song_data, score) for song_data, score in matches
            if song_data.get('title') != reference_song['song_data'].get('title')
        ]
        
        return filtered_matches[:max_results]
    
    def get_match_statistics(self):
        """Get statistics about the matching system"""
        total_songs = len(self.preprocessed_songs)
        total_ngrams = 0
        avg_ngrams_per_song = 0
        
        if total_songs > 0:
            for song_data in self.preprocessed_songs.values():
                for n, ngrams in song_data['ngrams'].items():
                    total_ngrams += len(ngrams)
            
            avg_ngrams_per_song = total_ngrams / total_songs
        
        return {
            'total_songs': total_songs,
            'total_ngrams': total_ngrams,
            'avg_ngrams_per_song': avg_ngrams_per_song,
            'similarity_threshold': self.similarity_threshold,
            'ngram_sizes': self.ngram_processor.n_sizes
        }
    
    def update_similarity_threshold(self, new_threshold):
        """Update similarity threshold for matching"""
        self.similarity_threshold = max(0.0, min(1.0, new_threshold))
    
    def add_song_to_corpus(self, song_data):
        """Add a new song to the corpus and preprocess it"""
        # Generate unique song ID
        artist = song_data.get('Artist', song_data.get('artist', 'unknown'))
        title = song_data.get('Title', song_data.get('title', 'untitled'))
        song_id = f"{artist}_{title}".lower()
        song_id = song_id.replace(' ', '_').replace("'", "")
        
        lyrics = song_data.get('Lyric') or song_data.get('lyric', '')

        if lyrics and lyrics.strip():
            # Generate n-grams for the new song
            ngrams_dict = self.ngram_processor.process_text_to_ngrams(lyrics)
            if ngrams_dict:
                ngram_counters = self.ngram_processor.ngrams_to_counter(ngrams_dict)
                
                # Normalize the song data
                normalized_song_data = {
                    'title': title,
                    'artist': artist,
                    'album': song_data.get('Album', song_data.get('album', '')),
                    'lyric': lyrics,
                    'original_data': song_data
                }
                
                self.preprocessed_songs[song_id] = {
                    'song_data': normalized_song_data,
                    'ngrams': ngrams_dict,
                    'ngram_counters': ngram_counters,
                    'keywords': self.ngram_processor.extract_keywords(lyrics)
                }
                
                return song_id
        
        return None