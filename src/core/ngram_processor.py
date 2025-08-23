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

    # ---------------- Text Preprocessing ----------------
    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would", "'m": " am",
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are"
        }
        for k, v in contractions.items():
            text = text.replace(k, v)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text

    def tokenize(self, text):
        if not text:
            return []
        return text.split()

    # ---------------- N-grams ----------------
    def generate_ngrams(self, tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def process_text_to_ngrams(self, text):
        tokens = self.tokenize(self.preprocess_text(text))
        if not tokens:
            return {}
        all_ngrams = {}
        for n in self.n_sizes:
            ngrams = self.generate_ngrams(tokens, n)
            if ngrams:
                all_ngrams[n] = ngrams
        return all_ngrams

    def ngrams_to_counter(self, ngrams_dict):
        return {n: Counter(ngrams) for n, ngrams in ngrams_dict.items()}

    # ---------------- Training ----------------
    def train_corpus(self, text):
        tokens = self.tokenize(self.preprocess_text(text))
        self.unigram_counts = Counter(tokens)
        self.total_unigrams = len(tokens)
        self.ngram_counts = {1: self.unigram_counts}
        for n in self.n_sizes:
            ngrams = self.generate_ngrams(tokens, n)
            self.ngram_counts[n] = Counter(ngrams)

    # ---------------- Interpolated Probability ----------------
    def interpolated_probability_dynamic(self, word, context, lambdas=None, debug=False):
        # Return tiny probability if model uninitialized
        if self.total_unigrams == 0:
            return 1e-12

        # Unigram with light additive smoothing for OOV safety
        vocab_size = max(len(self.unigram_counts), 1)
        alpha = 1e-3
        count_w = self.unigram_counts.get(word, 0)
        p1 = (count_w + alpha) / (self.total_unigrams + alpha * vocab_size)

        # Bigram
        p2 = 0.0
        if len(context) >= 1:
            prev1 = context[-1]
            denom = self.unigram_counts.get(prev1, 0)
            if denom > 0:
                p2 = self.ngram_counts.get(2, Counter()).get((prev1, word), 0) / denom

        # Trigram
        p3 = 0.0
        if len(context) >= 2:
            prev2 = tuple(context[-2:])
            denom = self.ngram_counts.get(2, Counter()).get(prev2, 0)
            if denom > 0:
                p3 = self.ngram_counts.get(3, Counter()).get(prev2 + (word,), 0) / denom

        # 4-gram
        p4 = 0.0
        if len(context) >= 3:
            prev3 = tuple(context[-3:])
            denom = self.ngram_counts.get(3, Counter()).get(prev3, 0)
            if denom > 0:
                p4 = self.ngram_counts.get(4, Counter()).get(prev3 + (word,), 0) / denom

        # Interpret lambdas
        if lambdas is None:
            weights = {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.3}
        elif isinstance(lambdas, dict):
            weights = {
                1: lambdas.get(1, 0.0),
                2: lambdas.get(2, 0.0),
                3: lambdas.get(3, 0.0),
                4: lambdas.get(4, 0.0)
            }
        else:
            if len(lambdas) == 4:
                weights = {1: lambdas[0], 2: lambdas[1], 3: lambdas[2], 4: lambdas[3]}
            elif len(lambdas) == 3:
                rem = max(0.0, 1.0 - sum(lambdas))
                weights = {1: rem, 2: lambdas[0], 3: lambdas[1], 4: lambdas[2]}
            elif len(lambdas) == 2:
                rem = max(0.0, 1.0 - sum(lambdas))
                weights = {1: rem, 2: lambdas[0], 3: lambdas[1], 4: 0.0}
            else:
                weights = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}

        # Only include up to (context length + 1) order
        max_order = min(1 + len(context), 4)
        probs = {1: p1, 2: p2, 3: p3, 4: p4}
        included_orders = [o for o in range(1, max_order + 1) if weights.get(o, 0.0) > 0]

        # Fallback to unigram if all weights were zeroed
        if not included_orders:
            included_orders = [1]

        # Renormalize weights
        total_w = sum(weights[o] for o in included_orders)
        if total_w <= 0:
            normalized = {o: 1.0 / len(included_orders) for o in included_orders}
        else:
            normalized = {o: weights[o] / total_w for o in included_orders}

        p = sum(normalized[o] * probs[o] for o in included_orders)

        if debug:
            dbg = {f'p{o}': probs[o] for o in [1, 2, 3, 4]}
            wdbg = {f'w{o}': normalized.get(o, 0.0) for o in [1, 2, 3, 4]}
            print(f"Word:'{word}', Context:{context[-3:]}, {dbg}, {wdbg}, Interp:{p:.8f}")

        return max(p, 1e-12)

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

    # ---------------- Similarity ----------------
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
        return len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0.0

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
        dot = sum(counter1[term] * counter2[term] for term in common_terms)
        mag1 = sum(v ** 2 for v in counter1.values()) ** 0.5
        mag2 = sum(v ** 2 for v in counter2.values()) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def combine_ngram_scores(self, scores_dict, weights=None):
        if not scores_dict:
            return 0.0
        if weights is None:
            weights = {2: 0.3, 3: 0.4, 4: 0.3, 1: 0.1}
        total_score = 0.0
        total_weight = 0.0
        for n, score in scores_dict.items():
            w = weights.get(n, 1.0)
            total_score += score * w
            total_weight += w
        return total_score / total_weight if total_weight > 0 else 0.0

    def extract_keywords(self, text, min_length=3, max_keywords=10):
        text = self.preprocess_text(text)
        tokens = self.tokenize(text)
        keywords = [t for t in tokens if len(t) >= min_length and t not in self.stop_words]
        freq = Counter(keywords)
        return [w for w, _ in freq.most_common(max_keywords)]

    def get_text_statistics(self, text):
        processed = self.preprocess_text(text)
        tokens = self.tokenize(processed)
        stats = {
            'original_length': len(text),
            'processed_length': len(processed),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'average_word_length': sum(len(w) for w in tokens) / len(tokens) if tokens else 0
        }
        ngrams_dict = self.process_text_to_ngrams(text)
        for n, ngrams in ngrams_dict.items():
            stats[f'{n}_gram_count'] = len(ngrams)
            stats[f'unique_{n}_grams'] = len(set(ngrams))
        return stats

    # ---------------- Next-word Prediction ----------------
    def predict_next_word_probabilities(self, context_words, top_k=10):
        if not self.unigram_counts:
            return []
        probs = []
        for w in self.unigram_counts.keys():
            p = self.interpolated_probability_dynamic(w, context_words)
            probs.append((w, p))
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

    def generate_text(self, seed_text, max_length=10):
        if not self.unigram_counts:
            return seed_text
        tokens = self.tokenize(self.preprocess_text(seed_text))
        output = tokens[:]
        for _ in range(max_length):
            candidates = self.predict_next_word_probabilities(output)
            if not candidates:
                break
            w, p = candidates[0]
            output.append(w)
        return ' '.join(output)
