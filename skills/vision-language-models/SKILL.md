# Vision-Language Models (VLMs): The Core Architecture

## Overview

**Vision-Language Models** teaches the architecture and techniques behind systems that understand both images and text. VLMs like GPT-4V, Claude, and LLaVA are the foundation of practical multimodal AI.

---

## What is a Vision-Language Model?

A VLM is a neural network that:

1. **Accepts image inputs** (single images or multiple images)
2. **Accepts text inputs** (questions, instructions, context)
3. **Produces text outputs** (descriptions, answers, reasoning)
4. **Understands relationships** between visual and textual content

```
Image(s) ┐
         ├─→ [VLM Neural Network] ─→ Text Output
Text     ┘   (typically LLM-based)   (generated tokens)
```

### Examples of Tasks

| Task | Input | Output |
|------|-------|--------|
| Image Captioning | Image | "A dog playing in a park" |
| Visual QA | Image + "What is the dog doing?" | "Playing fetch" |
| Document Understanding | Invoice image + "Extract date" | "2024-12-27" |
| Scene Understanding | Image | Detailed analysis of scene |
| Multi-image reasoning | Multiple images | Comparative analysis |

---

## Architecture: The LLaVA Model

LLaVA is one of the most instructive VLM architectures. Here's how it works:

### 1. Vision Encoder (Extract Image Information)

```
Input Image (480×480×3)
        ↓
[Vision Transformer (CLIP ViT-L)]
- Extract 576 patches (24×24 patches)
- Project each patch to 1024 dims
- Add positional embeddings
- Transformer processing
        ↓
Output: (576, 1024) = 576 visual tokens
```

**Why CLIP?**
- Already trained on image-text pairs (understands alignment)
- High quality embeddings
- No fine-tuning needed

### 2. Visual Projection (Convert Vision → Language Space)

```
Visual tokens (576, 1024)
        ↓
[Linear Projection Layer]
- Project 1024 → 4096 (language model dim)
        ↓
Projected visual tokens (576, 4096)
```

**Why?** Vision Transformer and LLM have different embedding spaces. This bridge connects them.

### 3. Language Model (Generate Output)

```
Projected visual tokens (576, 4096)
        +
Text tokens (e.g., "What's in this image?")
        ↓
[LLM Decoder (Llama 2 7B)]
- Attention over visual + text tokens
- Generate response one token at a time
        ↓
Output: "A dog running through a forest"
```

---

## Core Components

### 1. Vision Encoder Selection

```
┌─────────────────────────────────────┐
│ Vision Encoder Choices              │
├─────────────────────────────────────┤
│ CLIP (ViT-L/14) - 304M params      │
│  Pros: Image-text aligned           │
│  Cons: Standard resolution (336×336)│
│                                     │
│ DINOv2 (ViT-B) - 86M params        │
│  Pros: Very efficient               │
│  Cons: No explicit text alignment   │
│                                     │
│ ViT-Huge - 632M params             │
│  Pros: Highest quality              │
│  Cons: Very large, slow             │
│                                     │
│ OpenCLIP (ViT-G/14) - 1.4B params  │
│  Pros: Better alignment than CLIP  │
│  Cons: Massive                      │
└─────────────────────────────────────┘
```

### 2. Projection Strategy

**Simple Linear Projection** (LLaVA)
```python
visual_tokens (576, 1024)
    ↓
nn.Linear(1024, 4096)
    ↓
projected (576, 4096)
```
- Pros: Simple, fast
- Cons: Limited expressiveness

**MLP Projection** (BLIP-2)
```python
visual_tokens (576, 1024)
    ↓
nn.Sequential([
    nn.Linear(1024, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096)
])
    ↓
projected (576, 4096)
```
- Pros: More flexible
- Cons: Slightly slower

**Cross-Attention Projection** (Flamingo)
```python
visual_tokens (576, 1024) attend to
language tokens (text) to generate
contextual projections
    ↓
projected (576, 4096)
```
- Pros: Optimal, learns what's relevant
- Cons: Complex, slower

### 3. Language Model Selection

```
7B Models (Fast, Cost-Effective)
├─ Llama 2 7B
├─ Mistral 7B
└─ Phi 2

13B Models (Better Quality, Moderate Cost)
├─ Llama 2 13B
├─ Qwen 14B
└─ LLaMA 13B

70B Models (Best Quality, High Cost)
├─ Llama 2 70B
├─ Falcon 40B/180B
└─ GPT-4, Claude (closed-source)
```

---

## Building a VLM: Step-by-Step

### Step 1: Load Components

```python
from transformers import CLIPVisionModel, AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load vision encoder
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 2. Load language model
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
language_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Create projection
mm_projector = nn.Linear(1024, language_model.config.hidden_size)
```

### Step 2: Build the VLM

```python
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, language_model, mm_projector):
        super().__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.mm_projector = mm_projector
        
        self.vision_model.eval()  # Keep vision frozen during training
    
    def encode_image(self, image_tensor):
        """Convert image to visual tokens"""
        with torch.no_grad():
            visual_output = self.vision_model(image_tensor)
        
        # visual_output.last_hidden_state: (batch, 257, 1024)
        # 256 patch tokens + 1 class token
        visual_tokens = visual_output.last_hidden_state[:, 1:, :]  # Skip [CLS]
        
        # Project to language model space
        projected = self.mm_projector(visual_tokens)
        return projected  # (batch, 256, 4096)
    
    def forward(self, images, text_ids, text_attention_mask):
        """
        Args:
            images: (batch, 3, 336, 336)
            text_ids: (batch, seq_len)
            text_attention_mask: (batch, seq_len)
        """
        # Encode images
        visual_features = self.encode_image(images)  # (batch, 256, 4096)
        
        # Embed text
        text_embeddings = self.language_model.model.embed_tokens(text_ids)
        
        # Interleave visual and text
        # For simplicity: prepend visual features to text
        combined_embeddings = torch.cat([
            visual_features,  # (batch, 256, 4096)
            text_embeddings   # (batch, seq_len, 4096)
        ], dim=1)
        
        # Create attention mask
        visual_mask = torch.ones(
            images.shape[0], visual_features.shape[1],
            device=text_ids.device
        )
        combined_mask = torch.cat([visual_mask, text_attention_mask], dim=1)
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask
        )
        
        return outputs
```

### Step 3: Generate Responses

```python
def generate_response(vlm, image, question, tokenizer, max_length=256):
    """Generate a response to a visual question"""
    
    # Prepare image
    image_tensor = image_processor(image, return_tensors="pt").pixel_values
    image_tensor = image_tensor.to(vlm.device)
    
    # Prepare prompt
    prompt = f"<image>\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(vlm.device)
    
    # Generate
    with torch.no_grad():
        output_ids = vlm.language_model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            top_p=0.9,
            temperature=0.7,
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Usage
answer = generate_response(vlm, image, "What is in this image?", tokenizer)
print(answer)  # "A dog running through a forest"
```

---

## Training a VLM

### Approach 1: Full Fine-tuning (Expensive)
```
┌─────────────────────────────────────────┐
│ Freeze nothing, train everything        │
│ • Vision encoder: [YES] tuned              │
│ • Projector: [YES] tuned                   │
│ • Language model: [YES] tuned              │
├─────────────────────────────────────────┤
│ Pros: Best performance                  │
│ Cons: Expensive, risk of overfitting    │
│ GPU memory: ~48GB for 7B model           │
└─────────────────────────────────────────┘
```

### Approach 2: Projector-Only Fine-tuning (Recommended)
```
┌─────────────────────────────────────────┐
│ Freeze vision & language, tune projector│
│ • Vision encoder: [NO] frozen             │
│ • Projector: [YES] tuned                   │
│ • Language model: [NO] frozen             │
├─────────────────────────────────────────┤
│ Pros: Cheap, stable, fast               │
│ Cons: Limited customization             │
│ GPU memory: ~16GB for 7B model           │
└─────────────────────────────────────────┘
```

```python
# Freeze vision and language models
for param in vlm.vision_model.parameters():
    param.requires_grad = False
for param in vlm.language_model.parameters():
    param.requires_grad = False

# Only train projector
optimizer = torch.optim.AdamW(vlm.mm_projector.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    images, text_ids, labels = batch
    
    outputs = vlm(images, text_ids, attention_mask)
    loss = compute_loss(outputs.logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Approach 3: LoRA Fine-tuning (Balanced)
```
┌─────────────────────────────────────────┐
│ Add low-rank adapters to language model │
│ • Vision encoder: [NO] frozen             │
│ • Projector: [YES] tuned                   │
│ • Language model: [YES] LoRA (0.1% params)│
├─────────────────────────────────────────┤
│ Pros: Good performance, efficient       │
│ Cons: Slightly complex                  │
│ GPU memory: ~24GB for 7B model           │
└─────────────────────────────────────────┘
```

---

## Production Considerations

### 1. Image Resolution Trade-offs

```
224×224 (CLIP Default)
├─ Pros: Fast, cheap
├─ Cons: Loses detail
└─ Use: Quick responses, low-res content

336×336 (LLaVA Standard)
├─ Pros: Good balance
├─ Cons: Moderate speed
└─ Use: Most applications

672×672 (High-res)
├─ Pros: Detailed understanding
├─ Cons: 4x cost, 4x slower
└─ Use: Documents, fine text, technical diagrams
```

### 2. Handling Multiple Images

```python
def handle_multiple_images(vlm, images, question, tokenizer):
    """Process multiple images in one query"""
    
    # Process each image separately
    image_features_list = []
    for image in images:
        image_tensor = image_processor(image, return_tensors="pt").pixel_values
        visual_features = vlm.encode_image(image_tensor)
        image_features_list.append(visual_features)
    
    # Stack features
    combined_visual = torch.cat(image_features_list, dim=1)
    
    # Continue as normal
    prompt = f"<image1> <image2> ... Question: {question}"
    # ...
```

### 3. Caching for Efficiency

```python
class CachedVLM:
    def __init__(self, vlm):
        self.vlm = vlm
        self.image_cache = {}  # hash -> visual_features
    
    def encode_image_cached(self, image):
        """Cache visual encodings to avoid recomputation"""
        image_hash = hash_image(image)
        
        if image_hash not in self.image_cache:
            with torch.no_grad():
                self.image_cache[image_hash] = self.vlm.encode_image(image)
        
        return self.image_cache[image_hash]
```

---

## Key Decisions When Building a VLM

| Decision | Option A | Option B | Best For |
|----------|----------|----------|----------|
| Vision Encoder | CLIP | DINOv2 | CLIP (alignment) |
| Projector | Linear | MLP | Linear (simple) |
| Language Model | 7B | 70B | 7B (cost) |
| Training | Full | LoRA | LoRA (balance) |
| Resolution | 336×336 | 672×672 | 336×336 (default) |
| Multi-image | Sequential | Parallel | Depends on model |

---

## Common Issues & Solutions

### Issue: Poor Visual Understanding
**Symptom**: VLM hallucinates content not in image

```
Solution: Increase resolution
vlm_high_res = VLMHighRes(resolution=672)
```

### Issue: Slow Inference
**Symptom**: Response takes 10+ seconds

```
Solution: Use smaller language model
vlm_fast = VLM(language_model="phi-2")  # vs Llama 70B
```

### Issue: Overfitting on Custom Data
**Symptom**: Great on training data, fails on test data

```
Solution: Use projector-only fine-tuning
# Don't fine-tune language model itself
```

---

## Key Takeaways

1. **VLMs = Vision Encoder + Projection + Language Model**
2. **Projection bridges the two spaces** (vision → language)
3. **CLIP is the standard vision encoder** (good alignment)
4. **Projector-only fine-tuning is usually enough**
5. **Resolution matters** for detail-sensitive tasks
6. **Caching is critical** for production performance

---

## Next Steps

1. See [examples/vision-language-chat](../../examples/vision-language-chat) for working code
2. Study [cost-optimization](../cost-optimization) for production efficiency
3. Read [evaluation](../evaluation) for assessing VLM quality

---

## References

- LLaVA: Visual Instruction Tuning (Liu et al., 2023)
- BLIP-2: Bootstrapping Language-Image Pre-training (Li et al., 2023)
- Flamingo: a Visual Language Model (Alayrac et al., 2022)
- Learning Transferable Models for Multimodal Learning (Radford et al., 2021)
