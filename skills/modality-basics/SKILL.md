# Modality Basics

## Overview

**Modality Basics** teaches the fundamental concepts of different data types in AI systems and how they're processed differently. Before combining modalities, you must understand their unique characteristics, constraints, and representations.

---

## What are Modalities?

A **modality** is a distinct channel of information or mode of communication.

### The Core Modalities

| Modality | Native Format | Key Characteristic | Challenge |
|----------|---------------|-------------------|-----------|
| **Text** | Sequences of tokens/characters | Discrete, sequential | Variable length |
| **Image** | 2D pixel arrays (3 channels RGB) | Continuous, spatial | High dimensionality |
| **Audio** | 1D waveforms sampled over time | Continuous, temporal | Compression without loss |
| **Video** | 3D arrays (spatial + temporal) | Continuous, 4D data | Massive file sizes |
| **Code** | Structured text with execution semantics | Discrete + executable | Symbolic meaning |
| **Structured Data** | Tables, JSON, graphs | Discrete relationships | Heterogeneous formats |

---

## Modality Characteristics

### Dimensionality
- **Text**: Low (10k-100k vocab) but long sequences
- **Image**: High (e.g., 1920×1080×3 = 6.2M pixels)
- **Video**: Extreme (high-res video: 1-100GB per minute)
- **Audio**: Medium (16kHz = 16k samples/sec)

### Temporal Nature
- **Temporal**: Video, Audio, Time-series data
- **Atemporal**: Images, Static text

### Compression Ratio
- **Text**: Can achieve 10:1 compression easily
- **Image**: 2-50:1 depending on quality needs
- **Audio**: 10-100:1 with perceptual codecs
- **Video**: 100-1000:1 with modern codecs

### Processing Cost
```
Text:      $0.01/1M tokens
Image:     $0.03 per image (varies by resolution)
Video:     $0.10/min (sampling + processing)
Audio:     $0.02/min
```

---

## Representation Strategies

### Text
```
Raw Input: "A cat sitting on a mat"
Tokenization: [101, 1037, 3216, 4965, 2006, 1037, 13912]
Embedding: [0.23, -0.41, 0.67, ..., 0.12]  # 768 dims (BERT)
```

### Image
```
Raw Input: JPEG file (1920×1080)
Preprocessing: Resize to 224×224
Patch Tokenization: 196 patches × 768 dims (ViT)
Embedding: Single vector [0.12, -0.33, ...] or patch embeddings
```

### Audio
```
Raw Input: WAV file, 16kHz
Mel-Spectrogram: (n_mels=128, time_steps) log-scaled frequency representation
Embedding: Sequence of embeddings, one per time frame
```

### Video
```
Raw Input: 30fps, 1080p video clip
Frame Sampling: Extract 1 frame per second (30 frames for 30-sec clip)
Process Each Frame: Extract image embeddings
Temporal Aggregation: LSTM/Transformer over frame embeddings
Final Embedding: Single vector or temporal sequence
```

---

## Key Principles

### 1. **Every Modality Needs Preprocessing**
No raw pixels/waveforms go directly to LLMs. All must be tokenized/embedded first.

```
Image → Resize → Patch → Embed → LLM
Audio → Spectrogram → Embed → LLM
Video → Sample Frames → Embed Each → Aggregate → LLM
```

### 2. **Information Density Varies Wildly**
- A single image can contain as much semantic content as pages of text
- A 1-second audio clip might convey just 1-10 tokens of information

**Principle**: Different modalities need different "compression ratios"

### 3. **Loss is Modality-Specific**
- Image compression: Imperceptible loss at 10:1, obvious at 100:1
- Audio compression: Transparent at 16:1, artifacts at 128:1
- Text compression: Lossy compression breaks syntax

### 4. **Alignment is Hard**
A 10-minute video isn't just 600 individual images; it has:
- Continuity constraints (consecutive frames are similar)
- Motion patterns (objects move predictably)
- Audio synchronization (lips sync with sound)

Simply treating frames independently loses this structure.

### 5. **Cost Scales Differently**
```
Text:  Linear with token count
Image: Constant (usually) regardless of "detail"
Video: Linear with number of frames × image cost
Audio: Linear with duration
```

---

## Common Mistakes

❌ **Treating all modalities the same**
- Images aren't just "long sequences of tokens"
- Audio isn't just "fast text"

✅ **Understanding modality-specific properties** and designing accordingly

---

❌ **Naive concatenation**
```python
# DON'T DO THIS:
combined = np.concatenate([image_embedding, text_embedding])
```

✅ **Thoughtful fusion** (see fusion-strategies skill)

---

❌ **Ignoring temporal structure**
- Video frames aren't independent
- Audio samples have continuity

✅ **Preserving temporal relationships** in your embeddings

---

❌ **Over-compressing images**
- Resizing to 224×224 loses fine detail
- Sometimes need multiple resolutions

✅ **Context-aware preprocessing** (low-res for overview, high-res for detail)

---

## Practical Decision Tree

**What modalities do I need to process?**

```
┌─ Text only
│  └─ Use standard LLM tokenization
│
├─ Text + Image
│  └─ Use Vision Language Model (VLM)
│
├─ Text + Image + Audio
│  └─ Choose: Sequential or simultaneous processing?
│     ├─ Sequential: Process each modality independently
│     └─ Simultaneous: Use fusion architecture
│
├─ Video + Audio
│  └─ Must handle temporal alignment
│
└─ Code + Documentation + Examples
   └─ Combine code parsing + text understanding
```

---

## Working with Each Modality

### Text
**Strengths**: Discrete, lossless, compressible  
**Processing**: Tokenization → Embedding  
**Cost**: Cheapest per unit information  

**When to use**: Instructions, descriptions, code

### Image
**Strengths**: Rich spatial information, dense context  
**Processing**: Resize → Patch tokenization → Embedding  
**Cost**: Medium per image  

**When to use**: Visual context, charts, diagrams, photographs

### Audio
**Strengths**: Captures tone, emotion, speech prosody  
**Processing**: Spectrogram → Embedding  
**Cost**: Low per second  

**When to use**: Speech understanding, music, sound events

### Video
**Strengths**: Captures motion, action, temporal sequences  
**Processing**: Frame sampling → Image embedding → Temporal aggregation  
**Cost**: Highest (scales with frames)  

**When to use**: Action understanding, event detection, motion analysis

---

## Encoding Pipelines

### Vision Encoder Example
```
Input Image (JPEG)
  ↓ [Resize to 224×224]
  ↓ [Normalize RGB values]
  ↓ [Extract patches: 16×16 → 196 patches]
  ↓ [Project each patch to 768 dims]
  ↓ [Add positional embeddings]
Output: (196, 768) patch embeddings
  or
Output: (768,) pooled embedding
```

### Audio Encoder Example
```
Input Audio (WAV, 16kHz)
  ↓ [Compute Mel-Spectrogram: n_mels=128, hop_length=160]
  ↓ [Log scale the values]
  ↓ [Normalize (per frame mean=0, std=1)]
  ↓ [Embed each time-step: 128 → 256 dims]
  ↓ [Aggregate temporally (mean, attention, LSTM)]
Output: (256,) audio embedding
```

### Video Encoder Example
```
Input Video (MP4, 30fps, 1080p, 30 seconds)
  ↓ [Sample frames: every 1 second → 30 frames]
  ↓ [Process each frame with Image Encoder]
  ↓ [Get 30 × 768 embeddings]
  ↓ [Temporal aggregation: Transformer or LSTM]
  ↓ [Output temporal features or single summary]
Output: (30, 768) or (768,) depending on task
```

---

## Cost Calculation

### Example: Processing 1 hour of content

```
1 hour = 3600 seconds

TEXT:
- Average transcript: 5 words/sec = 18,000 words = 25,000 tokens
- Cost: $0.25 (at $10/1M tokens)

IMAGE:
- 100 screenshots from the video
- Cost: $3 (at $0.03/image)

AUDIO:
- 3600 seconds of audio
- Cost: $72 (at $0.02/min)

VIDEO:
- 3600 frames (1fps sampling)
- Cost: $108 (3600 × $0.03/frame)
```

**Takeaway**: Video processing is expensive. Use smart sampling.

---

## Decision Framework for Modality Selection

**Question 1**: Does this modality add unique information?
- If no, don't include it (reduces cost, complexity)
- If yes, include it

**Question 2**: Can I extract it from another modality?
- Video → extract text from OCR instead of video frames?
- Audio → extract text from speech-to-text instead of audio?

**Question 3**: What's my latency budget?
- Real-time: Process fewer frames, lower resolution
- Batch: Can process high-resolution, more frames

**Question 4**: What's my cost budget?
- Tight: Preprocess aggressively, compress modalities
- Loose: Higher resolution, more modalities

---

## Key Takeaways

1. **Each modality has unique characteristics** - don't treat them the same
2. **Preprocessing is essential** - raw data never goes directly to LLMs
3. **Information density varies** - images and video contain massive amounts of compressed information
4. **Cost varies widely** - choose modalities intentionally
5. **Temporal structure matters** - in video and audio
6. **Alignment is non-trivial** - combining modalities requires thought

---

## Next Steps

1. Read [embedding-spaces](../embedding-spaces) to learn how to represent different modalities in shared space
2. Study [fusion-strategies](../fusion-strategies) for how to combine them effectively
3. Explore architecture patterns like [vision-language-models](../vision-language-models)

---

## References

- Vision Transformer (Dosovitskiy et al., 2021)
- CLIP: Learning Transferable Models (Radford et al., 2021)
- HuBERT: Self-supervised audio representation (Hsu et al., 2021)
- TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)
