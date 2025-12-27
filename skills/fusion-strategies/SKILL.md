# Fusion Strategies: Combining Multiple Modalities

## Overview

**Fusion Strategies** teaches how to combine embeddings, features, and representations from different modalities into cohesive system outputs. The strategy you choose dramatically affects performance, cost, and latency.

---

## The Fusion Problem

You have multiple modality embeddings. Now what?

```
Image embedding:  (768,)
Text embedding:   (768,)  
Audio embedding:  (768,)

How do you combine them?
```

Three main approaches: **Early Fusion**, **Late Fusion**, and **Hybrid Fusion**.

---

## 1. Early Fusion

**Combine raw inputs before encoding.**

```
Input:
- Image (224×224×3)
- Text (tokens)
- Audio (16kHz waveform)
        ↓
    [Preprocess all]
        ↓
    [Concatenate or interleave]
        ↓
    [Single encoder]
        ↓
Output: (768,) fused embedding
```

### Advantages
[YES] Single unified representation
[YES] Can learn joint patterns early
[YES] Computationally efficient at inference
[YES] Natural alignment (encoder learns temporal sync)

### Disadvantages
[NO] Hard to preprocess (different input types)
[NO] Requires retraining if modality changes
[NO] Encoder must handle all modality types
[NO] Can be unstable (one modality drowns out others)

### Implementation

```python
class EarlyFusionModel(nn.Module):
    def __init__(self):
        # Convert everything to same format
        self.image_preprocessor = ImagePreprocessor()    # → (768,)
        self.text_preprocessor = TextPreprocessor()      # → (768,)
        self.audio_preprocessor = AudioPreprocessor()    # → (768,)
        
        # Single fusion encoder
        self.fusion_encoder = TransformerEncoder(
            input_dim=3*768,  # concatenate all three
            output_dim=768,
            num_layers=6
        )
    
    def forward(self, image, text, audio):
        # Preprocess each modality to same dimension
        img_features = self.image_preprocessor(image)
        txt_features = self.text_preprocessor(text)
        aud_features = self.audio_preprocessor(audio)
        
        # Concatenate
        combined = torch.cat([img_features, txt_features, aud_features], dim=-1)
        
        # Encode together
        fused_embedding = self.fusion_encoder(combined)
        
        return fused_embedding  # (768,)
```

### When to Use
- [YES] You have aligned, synchronized inputs (video + audio)
- [YES] You want lowest latency
- [YES] Modalities are never missing
- [NO] Inputs can be partial or misaligned

---

## 2. Late Fusion

**Encode each modality separately, then combine outputs.**

```
Image → Image Encoder → (768,)
                          ↓
                       [Fusion layer]
Text → Text Encoder → (768,)
                          ↓
                      Combined: (768,)
                          ↑
Audio → Audio Encoder → (768,)
```

### Advantages
[YES] Modular (each encoder independent)
[YES] Can use pretrained encoders
[YES] Handles missing modalities easily
[YES] More stable (each modality has own signal path)
[YES] Easy to add/remove modalities

### Disadvantages
[NO] Information loss (each encoding is independent)
[NO] Harder to learn cross-modal patterns
[NO] More computation (3 separate forward passes)
[NO] Fusion layer must learn what each modality means

### Implementation

```python
class LateFusionModel(nn.Module):
    def __init__(self):
        # Independent encoders
        self.image_encoder = VisionTransformer(output_dim=768)
        self.text_encoder = BERT(output_dim=768)
        self.audio_encoder = AudioTransformer(output_dim=768)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(3*768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768)
        )
    
    def forward(self, image, text, audio):
        # Encode separately
        img_emb = self.image_encoder(image)      # (768,)
        txt_emb = self.text_encoder(text)        # (768,)
        aud_emb = self.audio_encoder(audio)      # (768,)
        
        # Concatenate
        combined = torch.cat([img_emb, txt_emb, aud_emb], dim=-1)
        
        # Fuse
        fused = self.fusion(combined)  # (768,)
        
        return fused
    
    def forward_partial(self, **kwargs):
        """Handle missing modalities"""
        embeddings = []
        
        if 'image' in kwargs:
            embeddings.append(self.image_encoder(kwargs['image']))
        if 'text' in kwargs:
            embeddings.append(self.text_encoder(kwargs['text']))
        if 'audio' in kwargs:
            embeddings.append(self.audio_encoder(kwargs['audio']))
        
        combined = torch.cat(embeddings, dim=-1)
        
        # Fusion layer must handle variable sizes
        # Use adaptive pooling or padding
        fused = self.adaptive_fusion(combined)
        return fused
```

### When to Use
- [YES] You want modularity
- [YES] Some modalities may be missing
- [YES] You want to leverage pretrained models
- [YES] Modalities are independent
- [NO] Need tight cross-modal coupling

---

## 3. Hybrid Fusion

**Combine early and late fusion.**

```
Image → Early Fusion → (384,)
      ↗    (with audio)     ↘
Audio → Early Fusion ────→ Late Fusion → (768,)
                              ↑
                               ↓
Text → Text Encoder → (384,) ───
```

### Advantages
[YES] Best of both worlds
[YES] Early fusion captures tight sync (audio-visual)
[YES] Late fusion handles modularity (text)
[YES] Can use pretrained encoders
[YES] More flexible

### Disadvantages
[NO] More complex
[NO] More parameters
[NO] Harder to debug

### Implementation

```python
class HybridFusionModel(nn.Module):
    def __init__(self):
        # Early fusion: tightly coupled modalities
        self.image_encoder = VisionTransformer(output_dim=512)
        self.audio_encoder = AudioTransformer(output_dim=512)
        self.audiovisual_fusion = AudioVisualFusionModule(output_dim=384)
        
        # Late fusion: independent modality
        self.text_encoder = BERT(output_dim=384)
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(384 + 384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768)
        )
    
    def forward(self, image, text, audio):
        # Early fusion: audio-visual
        img_emb = self.image_encoder(image)
        aud_emb = self.audio_encoder(audio)
        av_fused = self.audiovisual_fusion(img_emb, aud_emb)  # (384,)
        
        # Late: text
        txt_emb = self.text_encoder(text)  # (384,)
        
        # Final fusion
        combined = torch.cat([av_fused, txt_emb], dim=-1)
        output = self.final_fusion(combined)  # (768,)
        
        return output
```

### When to Use
- [YES] Some modalities naturally go together (audio-visual)
- [YES] Others are independent (text)
- [YES] Want efficiency and modularity
- [YES] Complex applications

---

## 4. Cross-Attention Fusion

**Explicitly model modality interactions using attention.**

```
Image Features → Attention Head
                    ↓
                [Cross-Attention]
                    ↑
Text Features → Attention Head
```

### How It Works

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768):
        self.image_to_text = nn.MultiheadAttention(dim, num_heads=8)
        self.text_to_image = nn.MultiheadAttention(dim, num_heads=8)
        self.self_attention = nn.MultiheadAttention(dim, num_heads=8)
    
    def forward(self, image_emb, text_emb, audio_emb):
        # Image attends to text
        enhanced_image = self.image_to_text(
            query=image_emb,
            key=text_emb,
            value=text_emb
        )
        
        # Text attends to image
        enhanced_text = self.text_to_image(
            query=text_emb,
            key=image_emb,
            value=image_emb
        )
        
        # All together self-attend
        combined = torch.stack([enhanced_image, enhanced_text, audio_emb])
        fused = self.self_attention(
            query=combined,
            key=combined,
            value=combined
        )
        
        # Aggregate
        output = fused.mean(dim=0)  # (768,)
        return output
```

### Advantages
[YES] Explicitly models modality relationships
[YES] Learns what to attend to
[YES] Very flexible
[YES] State-of-the-art performance

### Disadvantages
[NO] Higher computation cost
[NO] Requires more training data
[NO] More parameters = more overfitting risk

### When to Use
- [YES] You have good training data
- [YES] Interaction patterns matter
- [YES] Latency is less critical
- [YES] You want best possible performance

---

## Comparison Table

| Strategy | Latency | Modularity | Performance | Complexity | Handles Missing |
|----------|---------|-----------|-------------|-----------|-----------------|
| Early | [FAST][FAST][FAST] | [STAR] | [STAR][STAR] | [SIMPLE] Simple | [NO] No |
| Late | [FAST][FAST] | [STAR][STAR][STAR] | [STAR][STAR] | [MEDIUM] Medium | [YES] Yes |
| Hybrid | [FAST][FAST] | [STAR][STAR] | [STAR][STAR][STAR] | [COMPLEX] Complex | [YES] Partial |
| Cross-Attention | [FAST] | [STAR][STAR][STAR] | [STAR][STAR][STAR][STAR] | [COMPLEX] Very Complex | [YES] Yes |

---

## Choosing Your Fusion Strategy

```
Do modalities arrive simultaneously/synchronized?
├─ YES → Consider Early Fusion for efficiency
└─ NO → Use Late Fusion for flexibility

Do you need to handle missing modalities?
├─ YES → Late or Cross-Attention
└─ NO → Any strategy works

Do you need maximum performance?
├─ YES → Cross-Attention Fusion
└─ NO → Late Fusion (good balance)

Do you have latency constraints?
├─ STRICT → Early Fusion
├─ MODERATE → Late Fusion
└─ LOOSE → Cross-Attention
```

---

## Practical Example: Implementing All Three

See [scripts/fusion_comparison.py](scripts/fusion_comparison.py) for complete working examples comparing all strategies on a benchmark task.

---

## Common Pitfalls

[NO] **Using Late Fusion when inputs are temporally synchronized**
- You'll lose temporal alignment
- [YES] Use Early or Hybrid instead

---

[NO] **Not normalizing embeddings before concatenation**
```python
# DON'T:
combined = torch.cat([image_emb, text_emb, audio_emb])

# DO:
combined = torch.cat([
    F.normalize(image_emb, p=2, dim=-1),
    F.normalize(text_emb, p=2, dim=-1),
    F.normalize(audio_emb, p=2, dim=-1)
])
```

---

[NO] **Treating all modalities equally**
- One modality might be noisier
- [YES] Learn modality-specific weights:
```python
weights = nn.Parameter(torch.ones(3))
weighted_combined = torch.cat([
    weights[0] * image_emb,
    weights[1] * text_emb,
    weights[2] * audio_emb
])
```

---

## Advanced: Gated Fusion

Give the model control over how much each modality contributes:

```python
class GatedFusion(nn.Module):
    def forward(self, image_emb, text_emb, audio_emb):
        combined = torch.cat([image_emb, text_emb, audio_emb], dim=-1)
        
        # Learn gates
        gate_image = torch.sigmoid(self.gate_image_net(combined))
        gate_text = torch.sigmoid(self.gate_text_net(combined))
        gate_audio = torch.sigmoid(self.gate_audio_net(combined))
        
        # Gated combination
        output = (
            gate_image * image_emb +
            gate_text * text_emb +
            gate_audio * audio_emb
        )
        
        return output
```

---

## Key Takeaways

1. **No universally best strategy** - choose based on your constraints
2. **Early Fusion** = efficiency but less flexible
3. **Late Fusion** = modularity, handles missing data
4. **Hybrid** = balance
5. **Cross-Attention** = best performance if you have compute
6. **Always normalize** before concatenation
7. **Modalities aren't equal** - consider weighting

---

## Next Steps

1. See [examples/vision-language-chat](../../examples/vision-language-chat) for Late Fusion in action
2. Study [evaluation](../evaluation) for how to benchmark fusion strategies
3. Read [cost-optimization](../cost-optimization) for efficient fusion at scale

---

## References

- Multimodal Machine Learning: A Survey and Taxonomy (Baltrušaitis et al., 2018)
- Transformers Can Do Bayesian Inference (Spantidakis et al., 2024)
- Vision-Language Pre-training (Alayrac et al., 2022)
