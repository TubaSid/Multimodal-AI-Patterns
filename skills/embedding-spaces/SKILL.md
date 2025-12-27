# Embedding Spaces: The Foundation of Multimodal AI

## Overview

**Embedding Spaces** teaches how to design and work with unified semantic spaces where different modalities can coexist and interact. This is the foundation of effective multimodal AI systems.

---

## The Core Challenge

Different modalities are born in different spaces:
- **Text** lives in discrete token space (vocab size: 50k)
- **Images** live in pixel space (dimensions: millions)
- **Audio** lives in waveform space (sample rate: 16-48kHz)

**Problem**: You can't directly compare an image embedding to a text embedding—they're in completely different spaces.

**Solution**: Project all modalities into a **shared semantic embedding space** where similar concepts cluster together, regardless of input modality.

---

## What is a Shared Embedding Space?

A shared space is a **d-dimensional vector space** where:

1. **Distance = Semantic Similarity**
   - Similar concepts are close together
   - Dissimilar concepts are far apart
   - This works across modalities

2. **Same Dimensionality**
   - Text embedding: (384,)
   - Image embedding: (384,)
   - Audio embedding: (384,)
   - ✅ Now comparable!

3. **Meaningful Operations**
   ```python
   # All valid in shared space:
   distance = cosine_similarity(image_emb, text_emb)
   nearest_images = retrieve_k_nearest(query_text_emb, all_image_embs)
   retrieved_audio = retrieve_k_nearest(query_image_emb, all_audio_embs)
   ```

---

## The CLIP Model: The Canonical Example

CLIP (Contrastive Language-Image Pre-training) is the gold standard for understanding shared spaces.

### How CLIP Works

```
Step 1: Encode Images
┌──────────────┐
│ Image        │
│ (224×224)    │
└──────┬───────┘
       ↓
┌──────────────────────────────────┐
│ Vision Transformer (ViT)         │
│ - Extract 196 patches (14×14)   │
│ - Project to 768 dims            │
│ - Global average pooling         │
└──────┬───────────────────────────┘
       ↓
   ┌────────┐
   │  (768,)│  Image Embedding
   └────────┘

Step 2: Encode Text
┌──────────────┐
│ Text:        │
│ "a photo    │
│  of a dog"   │
└──────┬───────┘
       ↓
┌──────────────────────────────────┐
│ Text Transformer (BERT-like)     │
│ - Tokenize (77 tokens max)      │
│ - Embed each token              │
│ - Take [CLS] token embedding    │
└──────┬───────────────────────────┘
       ↓
   ┌────────┐
   │  (768,)│  Text Embedding
   └────────┘

Step 3: Align in Shared Space
Both embeddings are (768,) → can compare!

Step 4: Training (Contrastive Learning)
- Batch of N images + N captions
- For each image: its caption should be close, others should be far
- For each caption: its image should be close, others should be far
- Loss: min(distance(img, correct_text)) + max(distance(img, wrong_text))
```

### Result
After training:
- Related images & texts cluster together
- Unrelated ones push apart
- ✅ Can retrieve images by text query or vice versa

---

## Creating Your Own Multimodal Space

### Step 1: Choose Base Encoders

```python
# Option A: Use existing models
image_encoder = timm.create_model('vit_base_patch16_224')
text_encoder = transformers.AutoModel.from_pretrained('bert-base')
audio_encoder = transformers.AutoModel.from_pretrained('wav2vec2-base')

# Option B: Build custom encoders
image_encoder = CustomVisionTransformer(dims=(3, 224, 224))
text_encoder = SimpleLSTM(input_size=vocab_size, hidden_size=768)
audio_encoder = Conv1DModel(input_channels=1, output_dims=768)
```

### Step 2: Project to Common Dimension

Encoders output different dimensions → normalize to shared dimension:

```python
class SharedSpaceProjector(nn.Module):
    def __init__(self, input_dims, shared_dim=768):
        super().__init__()
        self.projectors = nn.ModuleDict({
            'image': nn.Linear(input_dims['image'], shared_dim),
            'text': nn.Linear(input_dims['text'], shared_dim),
            'audio': nn.Linear(input_dims['audio'], shared_dim),
        })
        self.shared_dim = shared_dim
    
    def forward(self, modality, embedding):
        """Project embedding to shared space"""
        projected = self.projectors[modality](embedding)
        return F.normalize(projected, p=2, dim=-1)  # L2 normalize

# Usage
projector = SharedSpaceProjector({
    'image': 1024,  # ViT-B output
    'text': 768,    # BERT output
    'audio': 256,   # wav2vec output
})

image_in_shared_space = projector('image', image_emb)  # (768,)
text_in_shared_space = projector('text', text_emb)    # (768,)
audio_in_shared_space = projector('audio', audio_emb) # (768,)
```

### Step 3: Contrastive Training

```python
def contrastive_loss(embeddings, labels, temperature=0.07):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy)
    Standard loss for learning shared spaces
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    similarity_matrix = similarity_matrix / temperature
    
    # Positive pairs: same label
    # Negative pairs: different label
    
    loss = contrastive_loss_function(similarity_matrix, labels)
    return loss

# Training loop
for batch in training_data:
    images, texts, image_labels = batch
    
    # Encode
    image_embs = image_encoder(images)
    text_embs = text_encoder(texts)
    
    # Project to shared space
    image_shared = projector('image', image_embs)
    text_shared = projector('text', text_embs)
    
    # Loss: minimize distance between matched pairs
    loss = contrastive_loss(
        torch.cat([image_shared, text_shared]),
        torch.cat([image_labels, image_labels])
    )
    
    loss.backward()
    optimizer.step()
```

---

## Properties of Good Shared Spaces

### 1. **Isotropy**
Embeddings should be uniformly distributed in the space (not clustered in corners).

```python
# Check isotropy
cosine_sims = []
for emb1, emb2 in random_pairs:
    cosine_sims.append(cosine_similarity(emb1, emb2))

variance = np.var(cosine_sims)
# Good: variance near 0.33 (random distribution)
# Bad: variance near 0.0 (all embeddings similar)
```

### 2. **Modality Balance**
No single modality should dominate the space. Each should contribute equally.

```python
# Check balance
image_embs_std = np.std(image_embeddings, axis=0)
text_embs_std = np.std(text_embeddings, axis=0)
audio_embs_std = np.std(audio_embeddings, axis=0)

# Should be roughly equal across modalities
assert np.allclose(image_embs_std, text_embs_std, rtol=0.1)
```

### 3. **Semantic Alignment**
Modalities should cluster by semantic content, not by their source.

```python
# Validate alignment
image_of_dog_emb = encode_image(dog_image)
text_of_dog_emb = encode_text("a dog")
audio_of_dog_emb = encode_audio(dog_barking)

# All should be close
assert cosine_similarity(image_of_dog_emb, text_of_dog_emb) > 0.8
assert cosine_similarity(image_of_dog_emb, audio_of_dog_emb) > 0.7
```

### 4. **Stability**
Slight changes in input shouldn't drastically change embedding.

```python
# Perturb input slightly
image1_emb = encode_image(image)
image2_emb = encode_image(slightly_rotated_image)

# Should be very similar
assert cosine_similarity(image1_emb, image2_emb) > 0.95
```

---

## Different Space Designs

### Dense Spaces (Typical)
```
Dimensionality: 256-1024
Training: Contrastive learning
Use case: General-purpose multimodal retrieval
Pros: Works across modalities, compact
Cons: May lose fine-grained information
```

### Sparse Spaces
```
Dimensionality: 10k-100k
Training: Learned sparse codes
Use case: Interpretable multimodal reasoning
Pros: Interpretable (each dim = concept), modular
Cons: Less efficient, slower comparisons
```

### Hierarchical Spaces
```
Structure: Multiple embedding levels
- Low-level: Fine-grained features (high-dim)
- Mid-level: Semantic concepts (med-dim)
- High-level: Abstract meaning (low-dim)
Use case: Multi-granularity understanding
```

### Modality-Specific + Shared
```
Each modality has:
- Private space: Modality-specific information
- Shared space: Cross-modal information
Use case: Balance specificity with generality
```

---

## Common Issues & Solutions

### Issue 1: "Modality Collapse"
Problem: Text embeddings and image embeddings diverge too much.

```
Solution: Cross-modal contrastive loss
loss = sim(image, matched_text) - sim(image, unmatched_text)
```

### Issue 2: "Dimensionality Mismatch"
Problem: High-dimensional modalities dominate (curse of dimensionality).

```
Solution: Normalize embeddings carefully
embedding = F.normalize(embedding, p=2, dim=-1)
```

### Issue 3: "Semantic Drift"
Problem: Space isn't stable—nearby points in space aren't semantically similar.

```
Solution: Use stronger supervision
- Triplet loss (anchor, positive, negative)
- Hard negative mining
- Curriculum learning (easy → hard examples)
```

### Issue 4: "Posterior Collapse" (in VAE-based spaces)
Problem: Generator ignores latent space structure.

```
Solution: KL annealing + stronger prior
```

---

## Practical Example: Building an Image-Text-Audio Space

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import torchaudio

class MultimodalEmbedder:
    def __init__(self, shared_dim=768):
        # Encoders
        self.image_encoder = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.audio_encoder = self._build_audio_encoder()
        
        # Projectors
        self.image_proj = nn.Linear(512, shared_dim)  # CLIP image output
        self.text_proj = nn.Linear(768, shared_dim)   # BERT output
        self.audio_proj = nn.Linear(256, shared_dim)  # Audio encoder output
        
        self.shared_dim = shared_dim
    
    def embed_image(self, image):
        """Image → shared space"""
        emb = self.image_encoder(image)
        shared = self.image_proj(emb)
        return F.normalize(shared, p=2, dim=-1)
    
    def embed_text(self, text):
        """Text → shared space"""
        emb = self.text_encoder(text)
        shared = self.text_proj(emb)
        return F.normalize(shared, p=2, dim=-1)
    
    def embed_audio(self, audio):
        """Audio → shared space"""
        emb = self.audio_encoder(audio)
        shared = self.audio_proj(emb)
        return F.normalize(shared, p=2, dim=-1)
    
    def retrieve(self, query_embedding, candidates, k=5):
        """Find k nearest neighbors in shared space"""
        similarities = torch.matmul(query_embedding, candidates.T)
        top_k = torch.topk(similarities, k)
        return top_k.indices, top_k.values

# Usage
embedder = MultimodalEmbedder()

# Embed different modalities
image_emb = embedder.embed_image(image)
text_emb = embedder.embed_text(text)
audio_emb = embedder.embed_audio(audio)

# All are (768,) and comparable
assert image_emb.shape == text_emb.shape == audio_emb.shape

# Retrieve related content
similar_images = embedder.retrieve(text_emb, all_image_embeddings, k=10)
similar_audio = embedder.retrieve(image_emb, all_audio_embeddings, k=5)
```

---

## Evaluation of Shared Spaces

### Metrics

```python
# 1. Cross-modal retrieval accuracy
# Retrieve images by text query
# Measure: % of top-10 retrieved images semantically match query

# 2. Alignment score
# For matched pairs (image, caption), measure similarity
alignment = cosine_similarity(image_embs, text_embs).mean()

# 3. Modality balance
# Variance of embeddings should be similar across modalities

# 4. Semantic coherence
# Use human annotation to verify clusters make sense
```

---

## Key Takeaways

1. **Shared spaces are fundamental** - they enable cross-modal understanding
2. **Projection is key** - convert different modalities to same dimension
3. **Contrastive learning** - most effective training approach
4. **Normalization matters** - L2 normalization enables distance metrics
5. **Validation is essential** - check isotropy, balance, alignment
6. **Not all spaces are equal** - choose based on your task

---

## Next Steps

1. Study [fusion-strategies](../fusion-strategies) for combining embeddings
2. Explore [vision-language-models](../vision-language-models) for production VLMs
3. Read [cost-optimization](../cost-optimization) for efficient embedding storage

---

## References

- CLIP: Learning Transferable Models (Radford et al., 2021)
- ALIGN: Scaling Up Visual and Vision-Language Representation Learning (Jia et al., 2021)
- Contrastive Learning of General-Purpose Audio Representations (Chen et al., 2022)
- A Primer on Neural Network Architectures for Natural Language Processing (Goldberg, 2015)
