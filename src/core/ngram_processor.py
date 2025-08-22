import re
from collections import Counter

class NGramProcessor:
    """Handles n-gram generation and text preprocessing"""
    
    def __init__(self, n_sizes=[2, 3, 4]):
        """
        Initialize n-gram processor
        Args:
            n_sizes: List of n-gram sizes to generate (default: 2, 3, 4)
        """
        self.n_sizes = n_sizes
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'oh', 'yeah', 'hey', 'ah', 'uh'
        }
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text for n-gram generation
        Args:
            text: Raw text input
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "i'm": "i am", "you're": "you are",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle repeated characters (e.g., "sooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize preprocessed text into words
        Args:
            text: Preprocessed text
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        tokens = text.split()
        
        return tokens
    
    def generate_ngrams(self, tokens, n):
        """
        Generate n-grams from tokens
        Args:
            tokens: List of tokens
            n: N-gram size
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def process_text_to_ngrams(self, text):
        """
        Complete pipeline: text -> preprocessing -> tokenization -> n-grams
        Args:
            text: Raw text input
        Returns:
            Dictionary with n-gram sizes as keys and n-gram lists as values
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenize(processed_text)
        
        if not tokens:
            return {}
        
        # Generate n-grams for all specified sizes
        all_ngrams = {}
        for n in self.n_sizes:
            ngrams = self.generate_ngrams(tokens, n)
            if ngrams:
                all_ngrams[n] = ngrams
        
        return all_ngrams
    
    def ngrams_to_counter(self, ngrams_dict):
        """
        Convert n-grams dictionary to Counter objects
        Args:
            ngrams_dict: Dictionary with n-gram sizes and lists
        Returns:
            Dictionary with n-gram sizes and Counter objects
        """
        counters = {}
        for n, ngrams in ngrams_dict.items():
            counters[n] = Counter(ngrams)
        return counters
    
    def calculate_ngram_similarity(self, ngrams1, ngrams2, similarity_type='jaccard'):
        """
        Calculate similarity between two n-gram sets
        Args:
            ngrams1: First set of n-grams (Counter or dict)
            ngrams2: Second set of n-grams (Counter or dict)
            similarity_type: 'jaccard' or 'cosine'
        Returns:
            Similarity score (0-1)
        """
        if similarity_type == 'jaccard':
            return self._jaccard_similarity(ngrams1, ngrams2)
        elif similarity_type == 'cosine':
            return self._cosine_similarity(ngrams1, ngrams2)
        else:
            raise ValueError("Similarity type must be 'jaccard' or 'cosine'")
    
    def _jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity between two sets"""
        if isinstance(set1, Counter):
            set1 = set(set1.keys())
        if isinstance(set2, Counter):
            set2 = set(set2.keys())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, counter1, counter2):
        """Calculate cosine similarity between two Counter objects"""
        if not isinstance(counter1, Counter):
            counter1 = Counter(counter1)
        if not isinstance(counter2, Counter):
            counter2 = Counter(counter2)
        
        if not counter1 or not counter2:
            return 0.0
        
        # Get common terms
        common_terms = set(counter1.keys()).intersection(set(counter2.keys()))
        
        if not common_terms:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(counter1[term] * counter2[term] for term in common_terms)
        
        # Calculate magnitudes
        magnitude1 = sum(count ** 2 for count in counter1.values()) ** 0.5
        magnitude2 = sum(count ** 2 for count in counter2.values()) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def combine_ngram_scores(self, scores_dict, weights=None):
        """
        Combine scores from different n-gram sizes
        Args:
            scores_dict: Dictionary with n-gram sizes and their scores
            weights: Dictionary with weights for each n-gram size
        Returns:
            Combined weighted score
        """
        if not scores_dict:
            return 0.0
        
        if weights is None:
            # Default weights: higher n-grams get more weight
            weights = {2: 0.3, 3: 0.4, 4: 0.3}
        
        total_score = 0.0
        total_weight = 0.0
        
        for n, score in scores_dict.items():
            weight = weights.get(n, 1.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def extract_keywords(self, text, min_length=3, max_keywords=10):
        """
        Extract important keywords from text
        Args:
            text: Input text
            min_length: Minimum keyword length
            max_keywords: Maximum number of keywords to return
        Returns:
            List of keywords sorted by importance
        """
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        
        # Filter tokens
        keywords = [
            token for token in tokens 
            if len(token) >= min_length and token not in self.stop_words
        ]
        
        # Count frequency
        keyword_freq = Counter(keywords)
        
        # Get most common keywords
        return [word for word, freq in keyword_freq.most_common(max_keywords)]
    
    def get_text_statistics(self, text):
        """
        Get various statistics about the text
        Args:
            text: Input text
        Returns:
            Dictionary with text statistics
        """
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        
        stats = {
            'original_length': len(text),
            'processed_length': len(processed_text),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'average_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        }
        
        # Add n-gram counts
        ngrams_dict = self.process_text_to_ngrams(text)
        for n, ngrams in ngrams_dict.items():
            stats[f'{n}_gram_count'] = len(ngrams)
            stats[f'unique_{n}_grams'] = len(set(ngrams))
        
        return stats
    
    def find_similar_phrases(self, query_text, corpus_texts, threshold=0.3):
        """
        Find similar phrases in corpus using n-gram matching
        Args:
            query_text: Text to search for
            corpus_texts: List of texts to search in
            threshold: Minimum similarity threshold
        Returns:
            List of (text, similarity_score) tuples
        """
        query_ngrams = self.process_text_to_ngrams(query_text)
        query_counters = self.ngrams_to_counter(query_ngrams)
        
        similar_texts = []
        
        for text in corpus_texts:
            text_ngrams = self.process_text_to_ngrams(text)
            text_counters = self.ngrams_to_counter(text_ngrams)
            
            # Calculate similarity for each n-gram size
            scores = {}
            for n in self.n_sizes:
                if n in query_counters and n in text_counters:
                    score = self.calculate_ngram_similarity(
                        query_counters[n], 
                        text_counters[n],
                        'jaccard'
                    )
                    scores[n] = score
            
            if scores:
                combined_score = self.combine_ngram_scores(scores)
                if combined_score >= threshold:
                    similar_texts.append((text, combined_score))
        
        # Sort by similarity score (descending)
        similar_texts.sort(key=lambda x: x[1], reverse=True)
        return similar_texts