import re
from collections import Counter
import unicodedata

class NGramProcessor:
    """Handles n-gram generation, text preprocessing, similarity, and sequence probability"""
    
    def __init__(self, n_sizes=[2, 3, 4]):
        self.n_sizes = n_sizes
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'oh', 'yeah', 'hey', 'ah', 'uh'
        }
        self.ngram_counts = {}
        self.unigram_counts = Counter()
        self.total_unigrams = 0

    # ------------------- Preprocessing / Tokenization -------------------
    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # remove accents
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "i'm": "i am", "you're": "you are",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "they're": "they are"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text

    def tokenize(self, text):
        if not text:
            return []
        return text.split()

    # ------------------- N-grams -------------------
    def generate_ngrams(self, tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def process_text_to_ngrams(self, text):
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        if not tokens:
            return {}
        all_ngrams = {}
        for n in self.n_sizes:
            ngrams = self.generate_ngrams(tokens, n)
            if ngrams:
                all_ngrams[n] = ngrams
        return all_ngrams

    def ngrams_to_counter(self, ngrams_dict):
        counters = {}
        for n, ngrams in ngrams_dict.items():
            counters[n] = Counter(ngrams)
        return counters

    # ------------------- Training / Counting -------------------
    def train_corpus(self, text):
        tokens = self.tokenize(self.preprocess_text(text))
        self.unigram_counts = Counter(tokens)
        self.total_unigrams = len(tokens)
        self.ngram_counts = {}
        for n in self.n_sizes:
            ngrams = self.generate_ngrams(tokens, n)
            self.ngram_counts[n] = Counter(ngrams)

    # ------------------- Interpolated probability -------------------
    def interpolated_probability_dynamic(self, word, context, lambdas=None, debug=False):
        if lambdas is None:
            lambdas = [0.3, 0.4, 0.3]  # bigram, trigram, 4-gram

        p_uni = self.unigram_counts[word] / self.total_unigrams if self.total_unigrams > 0 else 1e-6

        p_bi = p_tri = p_quad = 0
        if len(context) >= 1:
            bg = tuple(context[-1:] + [word])
            count_bg = self.ngram_counts.get(2, Counter()).get(bg, 0)
            count_prev = self.ngram_counts.get(1, Counter()).get(tuple(context[-1:]), 0)
            p_bi = count_bg / count_prev if count_prev > 0 else 0
        if len(context) >= 2:
            tg = tuple(context[-2:] + [word])
            count_tg = self.ngram_counts.get(3, Counter()).get(tg, 0)
            count_prev = self.ngram_counts.get(2, Counter()).get(tuple(context[-2:]), 0)
            p_tri = count_tg / count_prev if count_prev > 0 else 0
        if len(context) >= 3:
            fg = tuple(context[-3:] + [word])
            count_fg = self.ngram_counts.get(4, Counter()).get(fg, 0)
            count_prev = self.ngram_counts.get(3, Counter()).get(tuple(context[-3:]), 0)
            p_quad = count_fg / count_prev if count_prev > 0 else 0

        if len(context) >= 3:
            p = lambdas[2]*p_quad + lambdas[1]*p_tri + lambdas[0]*p_bi + 0.01*p_uni
        elif len(context) == 2:
            p = lambdas[1]*p_tri + lambdas[0]*p_bi + 0.01*p_uni
        elif len(context) == 1:
            p = lambdas[0]*p_bi + 0.01*p_uni
        else:
            p = p_uni

        if debug:
            print(f"Word: '{word}', Context: {context}")
            print(f"  Uni: {p_uni:.6f}, Bi: {p_bi:.6f}, Tri: {p_tri:.6f}, Quad: {p_quad:.6f}")
            print(f"  Interpolated Prob: {p:.6f}\n")

        return max(p, 1e-6)

    def sequence_probability(self, words, lambdas=None, debug=False):
        prob = 1.0
        context = []
        for word in words:
            p = self.interpolated_probability_dynamic(word, context, lambdas, debug)
            prob *= p
            context.append(word)
            if len(context) > 3:
                context.pop(0)
        return prob

    # ------------------- Similarity -------------------
    def calculate_ngram_similarity(self, ngrams1, ngrams2, similarity_type='cosine'):
        if similarity_type == 'jaccard':
            return self._jaccard_similarity(ngrams1, ngrams2)
        elif similarity_type == 'cosine':
            return self._cosine_similarity(ngrams1, ngrams2)
        else:
            raise ValueError("Similarity type must be 'jaccard' or 'cosine'")

    def _jaccard_similarity(self, set1, set2):
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
        if not isinstance(counter1, Counter):
            counter1 = Counter(counter1)
        if not isinstance(counter2, Counter):
            counter2 = Counter(counter2)
        if not counter1 or not counter2:
            return 0.0
        common_terms = set(counter1.keys()).intersection(set(counter2.keys()))
        if not common_terms:
            return 0.0
        dot_product = sum(counter1[term]*counter2[term] for term in common_terms)
        magnitude1 = sum(count**2 for count in counter1.values())**0.5
        magnitude2 = sum(count**2 for count in counter2.values())**0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1*magnitude2)

    def combine_ngram_scores(self, scores_dict, weights=None):
        if not scores_dict:
            return 0.0
        if weights is None:
            weights = {2:0.3, 3:0.4, 4:0.3, 1:0.1}  # include unigram fallback weight
        total_score = 0.0
        total_weight = 0.0
        for n, score in scores_dict.items():
            weight = weights.get(n, 1.0)
            total_score += score * weight
            total_weight += weight
        return total_score / total_weight if total_weight > 0 else 0.0

    def extract_keywords(self, text, min_length=3, max_keywords=10):
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        keywords = [token for token in tokens if len(token) >= min_length and token not in self.stop_words]
        keyword_freq = Counter(keywords)
        return [word for word, freq in keyword_freq.most_common(max_keywords)]

    def get_text_statistics(self, text):
        processed_text = self.preprocess_text(text)
        tokens = self.tokenize(processed_text)
        stats = {
            'original_length': len(text),
            'processed_length': len(processed_text),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'average_word_length': sum(len(word) for word in tokens)/len(tokens) if tokens else 0
        }
        ngrams_dict = self.process_text_to_ngrams(text)
        for n, ngrams in ngrams_dict.items():
            stats[f'{n}_gram_count'] = len(ngrams)
            stats[f'unique_{n}_grams'] = len(set(ngrams))
        return stats

    # ------------------- Fixed find_similar_phrases -------------------
    def find_similar_phrases(self, query_text, corpus_texts, threshold=0.05):
        query_ngrams = self.process_text_to_ngrams(query_text)
        query_counters = self.ngrams_to_counter(query_ngrams)
        similar_texts = []
        for text in corpus_texts:
            text_ngrams = self.process_text_to_ngrams(text)
            text_counters = self.ngrams_to_counter(text_ngrams)
            scores = {}
            for n in self.n_sizes:
                if n in query_counters and n in text_counters:
                    score = self.calculate_ngram_similarity(query_counters[n], text_counters[n], 'cosine')
                    scores[n] = score
            # fallback to unigram cosine similarity if all n-grams are zero
            if not scores or all(v == 0 for v in scores.values()):
                uni_score = self._cosine_similarity(Counter(self.tokenize(query_text)),
                                                   Counter(self.tokenize(text)))
                scores[1] = uni_score
            if scores:
                combined_score = self.combine_ngram_scores(scores)
                if combined_score >= threshold:
                    similar_texts.append((text, combined_score))
        similar_texts.sort(key=lambda x: x[1], reverse=True)
        return similar_texts