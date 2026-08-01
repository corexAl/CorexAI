"""Dataset utilities for COREX — loading, tokenizing, and batching."""
import os
import random
from pathlib import Path
from typing import List, Dict

import numpy as np


def load_dataset(name: str = "wikitext", split: str = "train", max_samples=None):
    """Load text data for training.
    
    Priority order:
      1. Local text files in name directory
      2. HuggingFace datasets (c4, wikitext)  
      3. Fallback to synthetic corpus generation
      
    Args:
        name: dataset identifier (filename.txt, directory/, c4, wikitext)
        split: which split to use (for HF datasets)
        max_samples: cap on number of samples returned
        
    Returns:
        List[str]: list of text passages ready for tokenization
    """

    # Try loading from filesystem first
    if os.path.isfile(name):
        with open(name, "r", encoding="utf-8") as f:
            return [f.read()][:max_samples or 9999]
    
    if os.path.isdir(name):
        texts = []
        for fp in sorted(Path(name).glob("*.txt")):
            with open(fp, "r", encoding="utf-8") as f:
                texts.append(f.read())
        return texts[:max_samples or 9999]

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset as hf_load
        
        if name == "c4":
            data = hf_load("allenai/c4", "realnewslike", split=split)
            texts = [t for t in data["text"] if len(t.strip()) > 200]
            return texts[:max_samples or 9999]

        elif name == "wikitext":
            data = hf_load("wikitext", "wikitext-2-v1", split=split)
            texts = [t for t in data["text"] if len(t.strip()) > 50]
            return texts[:max_samples or 9999]

        elif name == "openwebtext":
            data = hf_load("Skylion007/openwebtext", split=split)
            texts = [t for t in data["text"] if len(t.strip()) > 100]
            return texts[:max_samples or 9999]

    except ImportError:
        pass  # datasets library not installed, will use synthetic fallback
    
    print(f"  ⚠ Using synthetic training data (datasets library not available)")
    return _generate_synthetic_corpus(max_samples or 5000)


def prepare_training_data(sequence_length: int = 64, is_eval: bool = False):
    """Convert raw text into fixed-length token sequences for masked LM training.
    
    Each sample contains (sequence_length+1) bytes encoded as integers in [0, 255].
    The model learns to predict byte N+1 given all previous bytes — this is 
    the core of autoregressive language modeling at the byte level.
    
    Uses a stride of sequence_length // 2 so that training sequences overlap,
    ensuring every byte participates in multiple training examples.
    
    Args:
        sequence_length: length of each context window (target = last token)
        is_eval: if True use fewer samples and non-overlapping windows
        
    Returns:
        List[dict]: each item has 'input_ids' (context tokens) and 'labels' (targets)
    """

    # Load corpus text
    dataset_name = "wikitext" if is_eval else "c4"
    raw_texts = load_dataset(dataset_name, max_samples=500 if is_eval else 2000)
    
    # Join all text into one stream
    full_text = "\n\n".join(raw_texts[:100])  # take first 100 texts
    
    # Encode to byte values
    token_ids = [b for b in full_text.encode("utf-8")]

    if is_eval:
        # Non-overlapping for consistent eval  
        stride = sequence_length
    else:
        # Overlapping for better data utilization
        stride = sequence_length // 2

    samples = []
    for i in range(0, len(token_ids) - sequence_length, stride):
        window = token_ids[i : i + sequence_length + 1]
        
        input_ids = [t % 512 for t in window[:-1]]   # context tokens (mod vocab_size)
        labels = [t % 512 for t in window[1:]]       # target tokens

        samples.append({"input_ids": input_ids, "labels": labels})

    return samples


def get_batch(samples, batch_size, device):
    """Randomly sample a batch from the dataset."""
    if len(samples) == 0:
        raise ValueError("Empty dataset")
        
    indices = np.random.choice(len(samples), min(batch_size, len(samples)), replace=False)

    return {
        "input_ids": torch.tensor([samples[i]["input_ids"] for i in indices]).to(device),
        "labels": torch.tensor([samples[i]["labels"] for i in indices]).to(device),
    }


def _generate_synthetic_corpus(max_samples: int = 5000):
    """Fallback synthetic corpus with diverse text styles.
    
    Generates programming, science, and general knowledge passages 
    to provide varied training data when no external dataset is available.
    """

    texts = []

    # Programming examples
    code_snippets = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nfor i in range(10):\n    print(fibonacci(i))",
        "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder(root):\n    if not root:\n        return []\n    return inorder(root.left) + [root.val] + inorder(root.right)",
        "import json\nimport requests\n\ndef fetch_data(url):\n    response = requests.get(url)\n    return response.json()\n\nprint(fetch_data('https://api.example.com/data'))",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
        "import torch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self, input_dim, output_dim):\n        super().__init__()\n        self.layer = nn.Linear(input_dim, output_dim)\n    \n    def forward(self, x):\n        return self.layer(x)",
    ]

    # Science content  
    science_text = [
        "The speed of light in a vacuum is approximately 299,792,458 meters per second. This fundamental constant, denoted by c, plays a crucial role in Einstein's theory of relativity and all of modern physics.",
        "Water consists of two hydrogen atoms bonded to one oxygen atom. The molecular structure is bent with a bond angle of approximately 104.5 degrees, giving water its unique polar properties and making it an excellent solvent.",
        "DNA contains the genetic instructions for the development and function of living organisms. It consists of two strands forming a double helix structure held together by hydrogen bonds between complementary base pairs: adenine with thymine, and guanine with cytosine.",
        "The mitochondrion is often called the powerhouse of the cell because it generates most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. The process of producing ATP from nutrients occurs through oxidative phosphorylation in the inner mitochondrial membrane.",
        "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It introduces concepts such as wave-particle duality, superposition, and quantum entanglement.",
    ]

    # General knowledge essays
    essay_texts = [
        "The development of artificial intelligence has progressed remarkably over the past decade. From rule-based expert systems to deep learning neural networks, AI has transformed numerous industries including healthcare, finance, education, and transportation. The key breakthrough came with the availability of large-scale datasets and powerful GPU compute.",
        "Machine learning algorithms can be broadly classified into supervised learning, unsupervised learning, and reinforcement learning. Supervised learning trains models on labeled examples, unsupervised learning discovers patterns in unlabeled data, and reinforcement learning optimizes agents through reward signals.",
        "Natural language processing involves teaching computers to understand, interpret, and generate human language. Key tasks include text classification, sentiment analysis, named entity recognition, machine translation, question answering, and text generation. Recent advances in transformer architectures have dramatically improved performance on these tasks.",
        "Deep neural networks have achieved remarkable success across diverse domains including computer vision, speech recognition, natural language processing, and game playing. These architectures learn hierarchical representations from raw data through multiple layers of nonlinear transformations, automatically discovering features at different levels of abstraction.",
    ]

    # Creative content  
    creative_text = [
        "The old lighthouse stood against the horizon, its beam cutting through the fog like a sword of light. Each night, the keeper would climb the spiral staircase and watch the sea churn below, wondering what storms still lay ahead and what tales the wind carried from distant shores.",
        "In the heart of the ancient forest, where sunlight filtered through the canopy in golden shafts, an old oak tree whispered stories to those who listened carefully enough. Its roots ran deep, connecting it to generations past, each ring a chapter in an unfolding saga of survival and growth.",
        "She opened the door and found herself standing at the crossroads of her life, each path promising something different: adventure on one side, safety on the other, or perhaps nothing at all. The choice was hers alone to make, and she knew that no matter which direction she chose, she would never look back.",
        "The city lights stretched out below like a sea of stars brought down to earth. Somewhere in that vast expanse of civilization lay countless stories waiting to unfold, each one unique and irreplaceable. A child was born in a hospital downtown. A business deal closed on the forty-second floor. And somewhere between them, a stranger's eyes met another stranger's across a crowded room.",
    ]

    # Fill up to max_samples with repeated cycles  
    all_texts = code_snippets + science_text + essay_texts + creative_text
    
    result = []
    for cycle in range(max(0, max_samples // len(all_texts) + 1)):
        # Shuffle within each cycle for variety
        shuffled = list(all_texts)
        random.shuffle(shuffled)
        result.extend(shuffled)

    return result[:max_samples]


if __name__ == "__main__":
    print("Testing dataset utilities...")
    
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
