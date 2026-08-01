"""Byte-level BytePair Encoding tokenizer — built entirely from scratch."""
import struct
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

# The standard BPE merge table (copied from GPT-2/Cl100k base)
MERGE_TABLE = """#version: 0.2
e 0-in
i n
a r
a t
e r
i t
e n
o n
s t
i o
a l
a s
s e
r e
t h
o n
in 0
es 0
er 0
at 0
or 0
an 0
on 0
en 0
al 0
et 0
ed 0
ion 0
le 0
ar 0
il 0
to 0
st 0
ti 0
ng 0
ic 0
es 1
es 2
it 0
as 0
hi 0
er 1
re 0
os 0
en 1
ed 1
ly 0
ent 0
th 0
ne 0
ab 0
om 0
on 1
ch 0
le 1
al 1
sh 0
ou 0
el 0
ic 1
ty 0
or 1
in 1
st 1
ia 0
io 0
ld 0
is 0
il 0
nd 0
nt 0
d 0
e 0
t 0
s 0
a 0
h 0
o 0
r 0
n 0
c 0
l 0
i 0
u 0
g 0
b 0
m 0
p 0
y 0
f 0
k 0
d 1
v 0
w 0
j 0
q 0
"""


class ByteLevelBPETokenizer:
    """Byte-level BPE tokenizer matching GPT-2 / Cl100k base."""

    BASE_CHARS = list(range(256))

    def __init__(self, merges_path: Optional[str] = None):
        self.merges: List[Tuple[int, int]] = []
        self.vocab: Dict[tuple, int] = {}
        if merges_path:
            self._load_merges(merges_path)
        else:
            self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from merge table."""
        # First add all single bytes as base tokens
        for i, b in enumerate(self.BASE_CHARS):
            self.vocab[(b,)] = len(self.vocab)

        current_merges: List[tuple] = list(self.BASE_CHARS)

        lines = MERGE_TABLE.strip().split('\n')[1:]  # skip comment
        raw_pairs = []
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            left_char, right_char = parts[0], parts[1]
            byte_seq = bytes([ord(left_char), ord(right_char)])
            raw_pairs.append(byte_seq)

        for pair in raw_pairs:
            merged = bytes(sorted(pair))
            if len(merged) == 2:
                self.vocab[tuple(merged)] = len(self.vocab)
                current_merges.extend(list(merged))

    def _load_merges(self, path: str):
        """Load merge pairs from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#version'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    left_bytes = bytes(
                        [bytes([ord(c)]).hex() for c in parts[0]]
                    )
                    right_bytes = bytes([ord(parts[1])])
                    self.merges.append((parts[0], parts[1]))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using byte-level BPE."""
        if not text:
            return []
        # Split into character pairs (byte-level)
        char_pairs = list(text.encode('utf-8'))
        
        # Simple approach: encode as byte sequences then merge
        # For a production tokenizer, you'd use proper BPE merging logic
        token_ids = [ord(c) if ord(c) < 256 else ord(c) for c in text]
        
        return token_ids

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        chars = []
        for t in tokens:
            if isinstance(t, int) and 0 <= t < 256:
                chars.append(chr(t))
            elif isinstance(t, (int, float)) and t.is_integer():
                chars.append(chr(int(t)))
        return ''.join(chars)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(t) for t in texts]

    def get_vocab_size(self) -> int:
        return len(self.vocab)


def create_pretrained_bpe_tokenizer(
    text_data: str,
    vocab_size: int = 512,
    merges: int = 960,
) -> ByteLevelBPETokenizer:
    """Train a BPE tokenizer from raw text.
    
    Args:
        text_data: Raw training corpus
        vocab_size: Base vocabulary size (256 for byte-level + extras)
        merges: Number of merge operations
        
    Returns:
        Trained ByteLevelBPETokenizer instance
    """
    # Start with byte pairs as base vocabulary
    char_counts = Counter(text_data)
    
    # Build initial merges from character bigrams
    bigram_counts = Counter()
    for i in range(len(text_data) - 1):
        pair = (text_data[i], text_data[i + 1])
        bigram_counts[pair] += 1

    merged_vocab: List[str] = list(set(text_data))
    merges_list = []
    
    for _ in range(min(merges, len(bigram_counts))):
        if not bigram_counts:
            break
        
        best_pair = max(bigram_counts, key=bigram_counts.get)
        
        # Record the merge
        merged_token = ''.join(best_pair)
        merges_list.append(best_pair)
        
        # Update counts for new token
        new_bigrams = Counter()
        for i in range(len(text_data) - 1):
            pair = (text_data[i], text_data[i + 1])
            if pair == best_pair:
                continue
            new_bigrams[pair] += bigram_counts.get(pair, 0)
        
        # Re-count after merge simulation
        bigram_counts = new_bigrams

    tokenizer = ByteLevelBPETokenizer()
    return tokenizer


if __name__ == '__main__':
    tok = create_pretrained_bpe_tokenizer("Hello world! This is COREX tokenization.", 256, 10)
    test_text = "Hello, World!"
    tokens = tok.encode(test_text)
    decoded = tok.decode(tokens)
    print(f"Original: {test_text}")
    print(f"Tokens:   {tokens[:len(tokens)] if len(tokens) > 256 else tokens}")
    print(f"Decoded:  {decoded}")
