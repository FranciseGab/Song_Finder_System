"""
Song Matching Module - Debug Version
Adds print statements to trace flow and values for verification
"""

from ngram_processor import NGramProcessor
import re
from collections import defaultdict

class SongMatcher:
    """Handles song matching using multiple search strategies (with debug output)"""

    def __init__(self, corpus_manager, similarity_threshold=0.15):
        print("[INIT] Initializing SongMatcher...")
        self.corpus_manager = corpus_manager
        self.ngram_processor = NGramProcessor()
        self.similarity_threshold = similarity_threshold
        self.preprocessed_songs = {}
        self.keyword_index = defaultdict(list)
        self.corpus_text = ""

        self.preprocess_corpus()
        self.train_ngram_model()
        print("[INIT DONE] Corpus processed and NGram model trained.")

    # ---------------- Corpus Preprocessing ----------------
    def preprocess_corpus(self):
        print("[PREPROCESS] Starting corpus preprocessing...")
        all_songs = self.corpus_manager.get_all_songs()
        processed_count = 0
        all_lyrics = []

        for song_id, song_data in all_songs.items():
            title = song_data.get('Title', song_data.get('title', 'Unknown'))
            artist = song_data.get('Artist', song_data.get('artist', 'Unknown'))
            lyrics = None
            for key in ['Lyric', 'lyric', 'Lyrics', 'lyrics', 'text', 'Text']:
                if key in song_data and song_data[key]:
                    lyrics = song_data[key]
                    break
            if not lyrics or not isinstance(lyrics, str) or not lyrics.strip():
                print(f"[SKIP] Song {title} by {artist} has no valid lyrics.")
                continue

            normalized_song_data = {
                'title': title,
                'artist': artist,
                'lyric': lyrics,
                'album': song_data.get('Album', song_data.get('album', '')),
                'original_data': song_data
            }

            all_text = f"{title} {artist} {lyrics}".lower()
            keywords = self._extract_keywords(all_text)

            self.preprocessed_songs[song_id] = {
                'song_data': normalized_song_data,
                'keywords': keywords
            }

            for keyword in keywords:
                self.keyword_index[keyword].append(song_id)

            all_lyrics.append(lyrics)
            processed_count += 1
            print(f"[PREPROCESS] Added song: {title} by {artist} with {len(keywords)} keywords.")

        self.corpus_text = " ".join(all_lyrics)
        print(f"[PREPROCESS DONE] Total songs processed: {processed_count}")

    def train_ngram_model(self):
        if self.corpus_text:
            print("[TRAIN] Training NGram model on corpus...")
            self.ngram_processor.train_corpus(self.corpus_text)
            print("[TRAIN DONE] Model trained successfully.")
        else:
            print("[TRAIN] No corpus text available, skipping training.")

    # ---------------- Keyword Extraction ----------------
    def _extract_keywords(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'a', 'an', 'is', 'was', 'are', 'were'}
        return list(set(word for word in words if len(word) > 2 and word not in stop_words))

    # ---------------- Main Match Finder ----------------
    def find_matches(self, query_text, max_results=10, use_ngram_scoring=True):
        print(f"[MATCH] Searching for query: '{query_text}'")
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            print("[MATCH] Invalid or empty query.")
            return []

        query_text = query_text.lower().strip()
        all_results = []

        # Exact matches
        for song in self._search_exact_title(query_text):
            print(f"[MATCH] Exact title match: {song['title']}")
            all_results.append((song, 1.0, "exact_title"))

        for song in self._search_exact_artist(query_text):
            print(f"[MATCH] Exact artist match: {song['artist']}")
            all_results.append((song, 0.9, "exact_artist"))

        for song, score in self._search_exact_lyrics_phrase(query_text):
            print(f"[MATCH] Exact lyrics phrase match: {song['title']} ({score:.3f})")
            all_results.append((song, score, "exact_lyrics"))

        for song, score in self._search_by_keywords_with_scoring(query_text):
            print(f"[MATCH] Keyword match: {song['title']} ({score:.3f})")
            all_results.append((song, score, "keywords"))

        for song, score in self._search_partial_with_scoring(query_text):
            print(f"[MATCH] Partial match: {song['title']} ({score:.3f})")
            all_results.append((song, score, "partial"))

        if use_ngram_scoring and self.ngram_processor.total_unigrams > 0:
            for song, score in self._search_by_integrated_ngram_similarity(query_text):
                print(f"[MATCH] NGram similarity: {song['title']} ({score:.3f})")
                all_results.append((song, score, "ngram_similarity"))

        # Deduplicate keeping highest score
        unique_results = {}
        for song_data, score, match_type in all_results:
            song_id = f"{song_data['title']}_{song_data['artist']}"
            if song_id not in unique_results or score > unique_results[song_id][1]:
                unique_results[song_id] = (song_data, score, match_type)

        final_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        filtered_results = [(song, score) for song, score, match_type in final_results if score >= self.similarity_threshold]
        print(f"[MATCH DONE] Found {len(filtered_results)} matches (after filtering).")
        return filtered_results[:max_results]

    # ---------------- Integrated Interpolation Similarity ----------------
    def _search_by_integrated_ngram_similarity(self, query_text):
        matches = []
        processed_query = self.ngram_processor.preprocess_text(query_text)
        query_tokens = self.ngram_processor.tokenize(processed_query)

        if len(query_tokens) < 1:
            return matches

        for song_id, data in self.preprocessed_songs.items():
            lyrics = data['song_data']['lyric']
            lyrics_processed = self.ngram_processor.preprocess_text(lyrics)
            lyrics_tokens = self.ngram_processor.tokenize(lyrics_processed)

            # Interpolation score
            interpolated_score = 0.0
            for i, token in enumerate(query_tokens):
                p = self.ngram_processor.interpolated_probability_dynamic(token, query_tokens[:i])
                interpolated_score += p
            interpolated_score /= max(len(query_tokens), 1)

            # Cosine similarity
            query_ngrams = self.ngram_processor.ngrams_to_counter(self.ngram_processor.process_text_to_ngrams(processed_query))
            lyrics_ngrams = self.ngram_processor.ngrams_to_counter(self.ngram_processor.process_text_to_ngrams(lyrics_processed))
            ngram_sims = []
            for n in self.ngram_processor.n_sizes:
                if n in query_ngrams and n in lyrics_ngrams:
                    sim = self.ngram_processor.calculate_ngram_similarity(query_ngrams[n], lyrics_ngrams[n], 'cosine')
                    if sim > 0:
                        ngram_sims.append(sim)
            avg_ngram_sim = sum(ngram_sims) / len(ngram_sims) if ngram_sims else 0

            # Sequence similarity
            seq_sim = self._calculate_sequence_similarity(query_tokens, lyrics_tokens)

            final_score = (interpolated_score * 0.3 + avg_ngram_sim * 0.5 + seq_sim * 0.2)
            if final_score >= self.similarity_threshold:
                print(f"[NGRAM] {data['song_data']['title']} - Score: {final_score:.4f} (Interp={interpolated_score:.4f}, Cosine={avg_ngram_sim:.4f}, Seq={seq_sim:.4f})")
                matches.append((data['song_data'], final_score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:20]

    # ---------------- Sequence Similarity ----------------
    def _calculate_sequence_similarity(self, query_tokens, lyrics_tokens):
        if len(query_tokens) < 2:
            return 0
        query_bigrams = set(self.ngram_processor.generate_ngrams(query_tokens, 2))
        lyrics_bigrams = set(self.ngram_processor.generate_ngrams(lyrics_tokens, 2))
        if not query_bigrams:
            return 0
        return len(query_bigrams.intersection(lyrics_bigrams)) / len(query_bigrams)

    # ---------------- Other Search Methods ----------------
    def _search_exact_lyrics_phrase(self, query_text):
        matches = []
        if len(query_text.strip().split()) < 2:
            return matches
        query_cleaned = self.ngram_processor.preprocess_text(query_text)
        for song_id, data in self.preprocessed_songs.items():
            lyrics_cleaned = self.ngram_processor.preprocess_text(data['song_data']['lyric'])
            if query_cleaned in lyrics_cleaned:
                match_ratio = len(query_cleaned) / len(lyrics_cleaned) if lyrics_cleaned else 0
                score = min(0.95, 0.7 + match_ratio * 0.25)
                matches.append((data['song_data'], score))
        return matches

    def _search_by_keywords_with_scoring(self, query_text):
        matches = []
        query_words = set(query_text.split())
        matched_songs = defaultdict(int)
        for word in query_words:
            for song_id in self.keyword_index.get(word, []):
                matched_songs[song_id] += 1
        for song_id, match_count in matched_songs.items():
            song_keywords = set(self.preprocessed_songs[song_id]['keywords'])
            intersection = len(query_words.intersection(song_keywords))
            union = len(query_words.union(song_keywords))
            similarity = intersection / union if union else 0
            if similarity >= 0.1:
                matches.append((self.preprocessed_songs[song_id]['song_data'], similarity))
        return matches

    def _search_partial_with_scoring(self, query_text):
        matches = []
        for song_id, data in self.preprocessed_songs.items():
            title_lower = data['song_data']['title'].lower()
            artist_lower = data['song_data']['artist'].lower()
            title_sim = self._calculate_string_similarity(query_text, title_lower)
            artist_sim = self._calculate_string_similarity(query_text, artist_lower)
            best_sim = max(title_sim, artist_sim)
            if best_sim >= 0.2:
                matches.append((data['song_data'], best_sim))
        return matches

    def _calculate_string_similarity(self, str1, str2):
        if str1 in str2 or str2 in str1:
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            return shorter / longer
        words1 = set(str1.split())
        words2 = set(str2.split())
        if not words1 or not words2:
            return 0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    # ---------------- Prediction / Completion ----------------
    def predict_next_words(self, input_text, top_k=10, min_probability=1e-6):
        if self.ngram_processor.total_unigrams == 0:
            return {"error": "N-gram model not trained"}
        processed_text = self.ngram_processor.preprocess_text(input_text)
        context = self.ngram_processor.tokenize(processed_text)
        if not context:
            return {"error": "No valid context provided"}
        word_probabilities = []
        for word in self.ngram_processor.unigram_counts.keys():
            prob = self.ngram_processor.interpolated_probability_dynamic(word, context)
            if prob >= min_probability:
                word_probabilities.append((word, prob))
        word_probabilities.sort(key=lambda x: x[1], reverse=True)
        return {"context": context, "predictions": word_probabilities[:top_k], "total_candidates": len(word_probabilities)}

    def interactive_text_completion(self, input_text, max_words=5, top_k=3):
        if self.ngram_processor.total_unigrams == 0:
            return {"error": "N-gram model not trained"}
        processed_text = self.ngram_processor.preprocess_text(input_text)
        current_tokens = self.ngram_processor.tokenize(processed_text)
        if not current_tokens:
            return {"error": "No valid starting context"}
        completions = []
        for i in range(max_words):
            context = current_tokens[-3:] if len(current_tokens) >= 3 else current_tokens
            predictions = self.predict_next_words(" ".join(context), top_k=top_k)
            if "error" in predictions or not predictions["predictions"]:
                break
            next_word, probability = predictions["predictions"][0]
            current_tokens.append(next_word)
            completions.append({"step": i+1, "context": context.copy(), "predicted_word": next_word, "probability": probability, "current_text": " ".join(current_tokens)})
        return {"original_text": input_text, "final_completion": " ".join(current_tokens), "steps": completions}

    def complete_song_lyric(self, partial_lyric, max_completions=3):
        if self.ngram_processor.total_unigrams == 0:
            return {"error": "N-gram model not trained"}
        results = []
        for i in range(max_completions):
            completion = self.interactive_text_completion(partial_lyric, max_words=6, top_k=5)
            if "error" not in completion:
                results.append(completion)
        return {"input": partial_lyric, "completions": results}

    # ---------------- Exact Title / Artist Search ----------------
    def _search_exact_title(self, query):
        return [data['song_data'] for song_id, data in self.preprocessed_songs.items() if data['song_data']['title'].lower() == query]

    def _search_exact_artist(self, query):
        return [data['song_data'] for song_id, data in self.preprocessed_songs.items() if data['song_data']['artist'].lower() == query]

    # ---------------- Add / Update Songs ----------------
    def add_song_to_corpus(self, song_data):
        title = song_data.get('Title', song_data.get('title', 'Unknown'))
        artist = song_data.get('Artist', song_data.get('artist', 'Unknown'))
        lyrics = None
        for key in ['Lyric', 'lyric', 'Lyrics', 'lyrics', 'text', 'Text']:
            if key in song_data and song_data[key]:
                lyrics = song_data[key]
                break
        if not lyrics or not isinstance(lyrics, str) or not lyrics.strip():
            return False
        song_id = f"{title}_{artist}"
        normalized_song_data = {
            'title': title,
            'artist': artist,
            'lyric': lyrics,
            'album': song_data.get('Album', song_data.get('album', '')),
            'original_data': song_data
        }
        self.preprocessed_songs[song_id] = {'song_data': normalized_song_data, 'keywords': self._extract_keywords(lyrics)}
        self.train_ngram_model()
        return True
