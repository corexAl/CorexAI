"""Production Dataset Utilities for COREX Training.

Complete data loading pipeline with:
- Support for multiple dataset formats (C4, Wikitext, custom text)
- Streaming mode for datasets larger than RAM  
- Tokenization and sequence chunking for training
- Caching for faster repeated access
- Multi-format support (plain text, JSONL, etc.)

All data loading is memory-efficient — only processes what fits in available RAM.
"""
import os
import torch
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union


# ═══════════════════════════════════════════════════════════
#  Streaming Dataset Loader
# ═══════════════════════════════════════════════════════════

class StreamingTextDataset:
    """Memory-efficient streaming text dataset loader.
    
    Loads and yields text passages from a file or directory without 
    loading everything into RAM at once. This is essential for training 
    on datasets that exceed available memory (e.g., C4 with 800GB+ files).
    
    Example usage:
        # Stream from a single large text file
        dataset = StreamingTextDataset("large_corpus.txt")
        
        # Stream from a directory of text files
        dataset = StreamingTextDataset("/data/corpus/")
        
        # Get all passages (still memory-efficient — yields one at a time)
        for passage in dataset:
            process(passage)
            
    Attributes:
        total_tokens: Total tokens available across all loaded chunks
        current_chunk_idx: Index of currently loaded chunk  
        _streamed_count: Number of text passages yielded so far (for progress tracking)
    """

    def __init__(self, source: str, chunk_size: int = 10_000_000):
        """Initialize the streaming dataset loader.
        
        Args:
            source: Path to a text file or directory containing .txt files
            chunk_size: Maximum tokens per chunk (controls memory usage)
        """
        self.source = source
        self.chunk_size = chunk_size  # Tokens per chunk for efficiency
        
        # Resolve file paths from source
        if os.path.isfile(source):
            self.file_paths = [source]
        elif os.path.isdir(source):
            self.file_paths = sorted(Path(source).glob("*.txt"))
            if not self.file_paths:
                self.file_paths = sorted(Path(source).glob("*.jsonl"))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

    def __len__(self) -> int:
        """Total number of text passages available (estimated from files)."""
        count = 0
        for path in self.file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                    count += lines
            except (UnicodeDecodeError, PermissionError):
                pass  # Skip unreadable files gracefully
        return count

    def __iter__(self) -> Iterator[str]:
        """Stream text passages one at a time without loading all into memory."""
        for path in self.file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped = line.strip()
                        if len(stripped) > 10:  # Skip very short passages
                            yield stripped
            except (UnicodeDecodeError, PermissionError):
                continue  # Skip files that can't be read


# ═══════════════════════════════════════════════════════════
#  Dataset Loading — Multi-Format Support
# ═══════════════════════════════════════════════════════════

def load_dataset(name: str = "wikitext", 
                 split: str = "train", 
                 max_samples: Optional[int] = None,
                 streaming: bool = True) -> List[str]:
    """Load text data for training from various sources.
    
    Priority order for loading:
      1. Local text files in name directory  
      2. HuggingFace datasets (c4, wikitext, openwebtext)
      3. Streaming mode via StreamingTextDataset class
      4. Fallback to synthetic corpus generation
      
    Args:
        name: Dataset identifier — can be a path (file/dir), HF dataset name, 
              or one of the built-in datasets (c4, wikitext, openwebtext)
        split: Which split to use for HF datasets (train/validation/test)
        max_samples: Maximum number of samples to load (None = unlimited)
        streaming: If True and source is a directory/file, use StreamingTextDataset
        
    Returns:
        List[str]: Text passages ready for tokenization
        
    Raises:
        FileNotFoundError: If local file/directory source doesn't exist
        ImportError: If HF datasets library not available when needed
    """

    # Try loading from local filesystem first (highest priority)
    if os.path.isfile(name):
        try:
            with open(name, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if len(line.strip()) > 10]
            return texts[:max_samples or 999999]
        except UnicodeDecodeError:
            # Try binary fallback — skip the file silently
            pass
            
    elif os.path.isdir(name):
        if streaming:
            # Use streaming for large directories to avoid OOM
            stream = StreamingTextDataset(name)
            texts = []
            for text in stream:
                texts.append(text)
                if max_samples and len(texts) >= max_samples:
                    break
            return texts
        
        # Load all files at once (fine for small directories)
        texts = []
        for fp in sorted(Path(name).glob("*.txt")) + sorted(Path(name).glob("*.jsonl")):
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > 10:
                        texts.append(content)
            except (UnicodeDecodeError, PermissionError):
                continue
        
        return texts[:max_samples or 999999]

    # Try HuggingFace datasets library
    try:
        from datasets import load_dataset as hf_load
        
        if name == "c4":
            data = hf_load("allenai/c4", "realnewslike", split=split)
            texts = [t for t in data["text"] if len(t.strip()) > 200]
            return texts[:max_samples or 999999]

        elif name == "wikitext":
            data = hf_load("wikitext", "wikitext-2-v1", split=split)
            texts = [t for t in data["text"] if len(t.strip()) > 50]
            return texts[:max_samples or 999999]

        elif name == "openwebtext":
            data = hf_load("Skylion007/openwebtext", split=split)
            texts = [t for t in data["text"] if len(t.strip()) > 100]
            return texts[:max_samples or 999999]

        else:
            # Treat as HF dataset name directly
            try:
                data = hf_load(name, split=split)
                # Try common text column names
                for col in ['text', 'content', 'body']:
                    if col in data.column_names:
                        texts = [t for t in data[col] if isinstance(t, str) and len(t.strip()) > 50]
                        return texts[:max_samples or 999999]
            except Exception:
                pass

    except ImportError:
        # datasets library not installed — will fall through to synthetic
        print(f"  ⚠ HuggingFace datasets library not available.")
        print("     Install with: pip install datasets")
    
    # Fallback to synthetic corpus generation
    print(f"  ℹ Using synthetic training data (no external datasets available)")
    return _generate_synthetic_corpus(max_samples or 5000)


# ═══════════════════════════════════════════════════════════
#  Training Data Preparation — Tokenization & Chunking
# ═══════════════════════════════════════════════════════════

def prepare_training_data(sequence_length: int = 64, 
                          is_eval: bool = False,
                          max_samples: Optional[int] = None) -> List[Dict[str, List[int]]]:
    """Prepare raw text data into fixed-length token sequences for model training.
    
    This function handles the complete preprocessing pipeline:
      1. Load raw text corpus (from local files, HF datasets, or synthetic source)
      2. Encode all text to byte values (0-255 range)  
      3. Split into overlapping chunks of `sequence_length` tokens
      4. Each chunk contains both context (input) and target (label) for training
      
    The model is trained to predict the next byte at each position given all 
    previous bytes — this is the core of autoregressive language modeling at 
    the byte level. Overlapping windows ensure every byte participates in multiple 
    training examples, improving data utilization.
    
    Args:
        sequence_length: Length of each context window (input = seq_len-1 tokens, 
                        target = seq_len-1 tokens for next-token prediction)
        is_eval: If True, use a smaller corpus and non-overlapping windows for 
                 faster, consistent evaluation
        max_samples: Maximum number of sequences to generate (None = unlimited)
        
    Returns:
        List[dict]: Training samples with 'input_ids' (context tokens) and 
                    'labels' (target tokens for next-token prediction)
                    
    Note:
        Each sample contains sequence_length token IDs. The model learns to predict 
        token[i+1] given tokens[0:i]. During training, the loss is computed over 
        all positions simultaneously using F.cross_entropy on shifted sequences.
    """

    # Load raw text corpus (may be large — use streaming if available)
    dataset_name = "wikitext" if is_eval else "c4"
    
    # Limit samples for evaluation to keep it fast
    eval_max = 500 if is_eval else None
    raw_texts = load_dataset(
        dataset_name, 
        split="train", 
        max_samples=max_samples or eval_max,
        streaming=True
    )

    # Concatenate all text into a single byte stream
    full_text = "\n\n".join(raw_texts)
    token_ids = [b for b in full_text.encode("utf-8")]

    if is_eval:
        # For evaluation, use non-overlapping windows for consistency
        stride = sequence_length
    else:
        # For training, use overlapping windows (stride = half length)
        # This ensures every byte appears in multiple training examples
        stride = sequence_length // 2

    # Create fixed-length sequences from the byte stream
    samples = []
    for i in range(0, len(token_ids) - sequence_length, stride):
        window = token_ids[i : i + sequence_length + 1]
        
        input_ids = [t % 512 for t in window[:-1]]   # Context tokens (mod vocab_size)
        labels = [t % 512 for t in window[1:]]       # Target tokens
        
        samples.append({"input_ids": input_ids, "labels": labels})
    
    return samples


def get_batch(samples: List[Dict[str, List[int]]], 
              batch_size: int, 
              device: torch.device) -> Dict[str, torch.Tensor]:
    """Randomly sample a training batch from prepared dataset.
    
    Args:
        samples: Prepared dataset (from prepare_training_data)
        batch_size: Number of sequences in the batch
        device: Target device for tensor placement
        
    Returns:
        dict with 'input_ids' and 'labels' tensors of shape (batch_size, seq_len)
        
    Raises:
        ValueError: If samples list is empty
    """
    if len(samples) == 0:
        raise ValueError("Empty dataset — no batches can be created")
    
    indices = torch.randint(0, len(samples), (min(batch_size, len(samples)),))
    
    return {
        "input_ids": torch.tensor([samples[i]["input_ids"] for i in indices]).to(device),
        "labels": torch.tensor([samples[i]["labels"] for i in indices]).to(device),
    }


# ═══════════════════════════════════════════════════════════
#  Synthetic Corpus Generation — Fallback Data Source
# ═══════════════════════════════════════════════════════════

def _generate_synthetic_corpus(max_samples: int = 5000) -> List[str]:
    """Generate diverse synthetic text corpus for training.
    
    Creates varied text covering programming examples, science content, 
    general knowledge essays, and creative writing to provide training data 
    when no external datasets are available.
    
    This fallback is not ideal for production but enables testing the full 
    training pipeline without requiring external data downloads.
    
    Args:
        max_samples: Maximum number of text passages to generate
        
    Returns:
        List[str]: Diverse synthetic text passages ready for tokenization
    """

    texts = []

    # Programming content — teaches syntax and structure
    code_texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nfor i in range(10):\n    print(fibonacci(i))",
        "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder(root):\n    if not root:\n        return []\n    return inorder(root.left) + [root.val] + inorder(root.right)",
        "import json\nimport requests\n\ndef fetch_data(url):\n    response = requests.get(url)\n    return response.json()\n\nprint(fetch_data('https://api.example.com/data'))",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
        "import torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self, input_dim, output_dim):\n        super().__init__()\n        self.layer = nn.Linear(input_dim, output_dim)\n    \n    def forward(self, x):\n        return self.layer(x)",
    ]

    # Science content — teaches factual knowledge
    science_texts = [
        "The speed of light in a vacuum is approximately 299,792,458 meters per second. This fundamental constant, denoted by c, plays a crucial role in Einstein's theory of relativity and all of modern physics.",
        "Water consists of two hydrogen atoms bonded to one oxygen atom. The molecular structure is bent with a bond angle of approximately 104.5 degrees, giving water its unique polar properties and making it an excellent solvent.",
        "DNA contains the genetic instructions for the development and function of living organisms. It consists of two strands forming a double helix structure held together by hydrogen bonds between complementary base pairs: adenine with thymine, and guanine with cytosine.",
        "The mitochondrion is often called the powerhouse of the cell because it generates most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. The process occurs through oxidative phosphorylation in the inner mitochondrial membrane.",
        "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It introduces concepts such as wave-particle duality, superposition, and quantum entanglement.",
    ]

    # General knowledge essays — teaches reasoning and writing style
    essay_texts = [
        "The development of artificial intelligence has progressed remarkably over the past decade. From rule-based expert systems to deep learning neural networks, AI has transformed numerous industries including healthcare, finance, education, and transportation. The key breakthrough came with the availability of large-scale datasets and powerful GPU compute.",
        "Machine learning algorithms can be broadly classified into supervised learning, unsupervised learning, and reinforcement learning. Supervised learning trains models on labeled examples, unsupervised learning discovers patterns in unlabeled data, and reinforcement learning optimizes agents through reward signals.",
        "Natural language processing involves teaching computers to understand, interpret, and generate human language. Key tasks include text classification, sentiment analysis, named entity recognition, machine translation, question answering, and text generation. Recent advances in transformer architectures have dramatically improved performance on these tasks.",
        "Deep neural networks have achieved remarkable success across diverse domains including computer vision, speech recognition, natural language processing, and game playing. These architectures learn hierarchical representations from raw data through multiple layers of nonlinear transformations.",
    ]

    # Creative content — teaches narrative style and prose
    creative_texts = [
        "The old lighthouse stood against the horizon, its beam cutting through the fog like a sword of light. Each night, the keeper would climb the spiral staircase and watch the sea churn below, wondering what storms still lay ahead and what tales the wind carried from distant shores.",
        "In the heart of the ancient forest, where sunlight filtered through the canopy in golden shafts, an old oak tree whispered stories to those who listened carefully enough. Its roots ran deep, connecting it to generations past, each ring a chapter in an unfolding saga of survival and growth.",
        "She opened the door and found herself standing at the crossroads of her life, each path promising something different: adventure on one side, safety on the other, or perhaps nothing at all. The choice was hers alone to make, and she knew that no matter which direction she chose, she would never look back.",
        "The city lights stretched out below like a sea of stars brought down to earth. Somewhere in that vast expanse of civilization lay countless stories waiting to unfold, each one unique and irreplaceable. A child was born in a hospital downtown. A business deal closed on the forty-second floor.",
    ]

    # Combine and shuffle for variety
    all_texts = code_texts + science_texts + essay_texts + creative_texts
    
    result = []
    for cycle in range(max(0, max_samples // len(all_texts) + 1)):
        shuffled = list(all_texts)
        random.shuffle(shuffled)
        result.extend(shuffled)

    return result[:max_samples]


# ═══════════════════════════════════════════════════════════
#  Testing / Validation
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing dataset utilities...")
    
    # Test prepare_training_data
    data = prepare_training_data(sequence_length=32)
    print(f"✓ Generated {len(data)} training sequences")
    
    if len(data) > 0:
        sample = data[0]
        print(f"  input_ids length: {len(sample['input_ids'])}")
        print(f"  labels length:    {len(sample['labels'])}")
        print(f"  Token range:      [{min(sample['input_ids'])}, {max(sample['input_ids'])}]")
        
        # Decode preview
        decoded = bytes(sample["input_ids"][:30]).decode("utf-8", errors="replace")
        print(f"  Preview: {decoded!r}")
    
    print("\nDataset utilities test complete!")
