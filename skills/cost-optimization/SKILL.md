# Cost Optimization: Making Multimodal AI Affordable

## Overview

**Cost Optimization** teaches techniques for reducing API costs, inference latency, and infrastructure expenses while maintaining quality. At scale, this is the difference between profitable and unsustainable.

---

## The Cost Problem

Multimodal AI is **expensive**. Processing images, video, and audio adds up fast.

### Real-World Cost Example

**Processing 1 million customer support tickets (each: 1 image + 200 word transcript)**

```
Naive approach:
├─ Image processing: 1M × $0.03 = $30,000
├─ Audio transcription: 1M × $0.01 = $10,000
├─ LLM analysis: 1M × $0.001 = $1,000
└─ Total: $41,000 per run

Optimized approach:
├─ Image caching (85% hit rate): $4,500
├─ Voice cache: $0
├─ LLM with routing (50% simpler): $500
└─ Total: $5,000 per run
├─ Savings: 88% ($36,000)
```

---

## 1. Image Optimization

### 1.1 Compression Strategies

#### Strategy A: Adaptive Resolution

```python
def optimize_image_resolution(image, task_type):
    """Choose resolution based on task requirements"""
    
    if task_type == "ocr" or task_type == "fine_text":
        # Need high resolution for text extraction
        return resize_image(image, target_size=1024)
    
    elif task_type == "scene_understanding":
        # Standard resolution is fine
        return resize_image(image, target_size=512)
    
    elif task_type == "classification":
        # Low resolution is sufficient
        return resize_image(image, target_size=256)
    
    elif task_type == "thumbnail":
        return resize_image(image, target_size=128)

# Cost Impact
# 1024px = $0.03
# 512px = $0.025 (17% cheaper)
# 256px = $0.02 (33% cheaper)
```

#### Strategy B: Quality Reduction

```python
def compress_image(image, quality_target=0.8):
    """Reduce file size without perceptual quality loss"""
    
    from PIL import Image
    import io
    
    # JPEG compression
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    
    # WebP compression (better, newer)
    buffer = io.BytesIO()
    image.save(buffer, format='WEBP', quality=85)
    
    return buffer.getvalue()

# Comparison
# PNG (original): 2.5 MB
# JPEG (quality=85): 200 KB (92% reduction)
# WebP (quality=85): 150 KB (94% reduction)
```

#### Strategy C: Content-Aware Cropping

```python
def smart_crop_image(image, task="general"):
    """Remove irrelevant content before sending to API"""
    
    # Detect region of interest
    faces = detect_faces(image)
    objects = detect_objects(image)
    text_regions = detect_text(image)
    
    # For face recognition, crop to faces + context
    if task == "face_recognition":
        bounding_box = expand_bbox(faces[0], expand_ratio=1.5)
        return crop_image(image, bounding_box)
    
    # For document analysis, crop to document only
    if task == "ocr":
        document_bbox = detect_document(image)
        return crop_image(image, document_bbox)
    
    return image

# Cost Impact
# Full document (A4 scan): $0.03
# Cropped to text region: $0.01 (67% cheaper)
```

### 1.2 Batching & Caching

```python
class ImageCache:
    def __init__(self, cache_size_gb=10):
        self.cache = {}
        self.cache_size = cache_size_gb * 1e9
        self.current_size = 0
    
    def get_or_encode(self, image_path, encoder_fn):
        """Cache image encodings to avoid reprocessing"""
        
        image_hash = self.hash_image(image_path)
        
        if image_hash in self.cache:
            return self.cache[image_hash]
        
        # Process and cache
        embedding = encoder_fn(image_path)
        self.cache[image_hash] = embedding
        
        return embedding
    
    def hash_image(self, image_path):
        """Create deterministic hash"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

# Usage
image_cache = ImageCache()

# First call: ~$0.03
embedding1 = image_cache.get_or_encode('cat.jpg', encode_image)

# Second call: $0 (cached)
embedding2 = image_cache.get_or_encode('cat.jpg', encode_image)

# Expected cache hit rate: 60-85%
# Cost savings: 60-85%
```

---

## 2. Audio Optimization

### 2.1 Smart Sampling

```python
def optimize_audio(audio_file, target_sample_rate=16000):
    """Reduce audio file size while preserving information"""
    
    import librosa
    
    # Original: 44.1 kHz (high quality)
    # Optimized: 16 kHz (sufficient for speech)
    audio, sr = librosa.load(audio_file, sr=target_sample_rate)
    
    # 44.1 kHz file: 100 MB per hour
    # 16 kHz file: 36 MB per hour
    # Savings: 64%
    
    return audio, target_sample_rate

def extract_speech_regions(audio, sr):
    """Only process regions with speech, skip silence"""
    
    import librosa
    
    # Detect speech/silence
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    db = librosa.power_to_db(S)
    
    # Find speech frames
    speech_mask = db.mean(axis=0) > threshold
    
    # Extract only speech regions
    speech_audio = audio[speech_mask]
    
    # If 40% of audio is silence, skip it
    # Cost reduction: 40%
    
    return speech_audio
```

### 2.2 Smart Transcription

```python
class SmartTranscriber:
    def transcribe_optimal(self, audio_file):
        """Choose transcription model based on audio length"""
        
        duration = get_audio_duration(audio_file)
        
        if duration < 10:
            # Short clip: Use Whisper API
            # Cost: $0.06/min
            return whisper_api(audio_file)
        
        elif duration < 300:
            # Medium: Use local Whisper (one-time cost)
            # Cost: $0 (just compute)
            return local_whisper(audio_file)
        
        else:
            # Long: Chunk and process locally
            # Cost: $0 (just compute)
            return chunk_and_transcribe_local(audio_file)

# Cost comparison for 1 hour of audio
# API-only: $3.60
# Hybrid: $0.15 (96% savings)
```

---

## 3. Video Optimization

### 3.1 Intelligent Frame Sampling

```python
def sample_video_frames(video_file, strategy="adaptive"):
    """Extract frames only where they add value"""
    
    if strategy == "uniform":
        # Sample every Nth frame
        # 30fps video: sample every 30 frames = 1 frame/sec
        frames = extract_frames(video_file, fps=1)
        # 1 min video: 60 frames × $0.03 = $1.80
    
    elif strategy == "adaptive":
        # Sample more when content changes
        all_frames = extract_all_frames(video_file)
        
        # Compute optical flow (motion detection)
        key_frames = []
        for i in range(len(all_frames)):
            motion = optical_flow(all_frames[i], all_frames[i+1])
            
            if motion > threshold:
                key_frames.append(all_frames[i])
        
        # High-motion video: maybe 2-3 fps
        # Low-motion video: maybe 0.5 fps
        # 1 min video: ~30-50 frames × $0.03 = $0.90-1.50
        # Savings: 17-50%
        return key_frames
    
    elif strategy == "scene_change":
        # Sample only at scene cuts
        frames = extract_scene_changes(video_file)
        # Commercial video: maybe 10-20 frames/min
        # 1 min: 10-20 frames × $0.03 = $0.30-0.60
        # Savings: 67-85%
        return frames

# Cost comparison for 10 min video
# All frames (300fps): 3600 frames × $0.03 = $108
# Uniform (1fps): 600 frames × $0.03 = $18
# Adaptive: 400 frames × $0.03 = $12
# Scene changes: 150 frames × $0.03 = $4.50
```

### 3.2 Pre-processing to Reduce Compute

```python
def preprocess_video(video_file):
    """Compress video before frame extraction"""
    
    import subprocess
    
    # H.264 compression (lower bitrate)
    # Original: 500 Mbps
    # Compressed: 50 Mbps (90% reduction)
    
    subprocess.run([
        'ffmpeg',
        '-i', video_file,
        '-c:v', 'libx264',
        '-crf', '28',  # Quality: 0-51 (lower=better)
        '-preset', 'fast',
        'output_compressed.mp4'
    ])
    
    # Now extract frames from smaller file
    # 10x less I/O
```

---

## 4. LLM Cost Reduction

### 4.1 Model Selection

```python
# Cost per 1M tokens (input)

Tier 1: Ultra-cheap (< $0.10/1M)
├─ Phi-2 (local): $0
├─ Qwen 1.8B: ~$0.05
└─ Mistral 7B: $0.07

Tier 2: Mid-range ($0.10-$1.00)
├─ Claude 3 Haiku: $0.25
├─ GPT-4 Turbo: $0.01 (input)
└─ Llama 2 70B (hosted): $0.50

Tier 3: Premium ($1.00-$5.00)
├─ Claude 3 Opus: $3.00
├─ GPT-4V: $0.03 per image + $0.01/token
└─ Custom fine-tuned models

# Decision framework
if response_time < 5 seconds:
    use GPT-4 Turbo  # Fastest
elif accuracy_critical:
    use Claude 3 Opus  # Most accurate
elif cost_critical:
    use local Mistral 7B or Phi-2  # Cheapest
```

### 4.2 Prompt Optimization

```python
def optimize_prompt(task):
    """Reduce token count without losing quality"""
    
    # Verbose prompt (500 tokens)
    verbose = """
    You are an expert in analyzing customer support tickets.
    Please carefully analyze the following support ticket and 
    determine:
    1. The primary issue
    2. The severity (low, medium, high)
    3. The recommended resolution
    ...
    """
    
    # Optimized prompt (200 tokens)
    optimized = """
    Analyze this ticket. Return JSON:
    {"issue": "", "severity": "low|med|high", "resolution": ""}
    """
    
    # Cost: 500 tokens vs 200 tokens = 60% savings
    return optimized

def few_shot_optimization(examples):
    """Choose examples strategically to fit budget"""
    
    # All examples: 2000 tokens
    # Selected examples (semantic similarity): 800 tokens
    
    selected = select_diverse_examples(examples, budget=800)
    return selected

# Tool use optimization
def use_tools_to_reduce_tokens():
    """Delegate work to tools instead of LLM processing"""
    
    # Verbose: LLM processes entire document
    # 10,000 tokens for analysis
    
    # Optimized: Tool extracts data, LLM analyzes summary
    # Extract with regex: 0 tokens
    # Analyze summary: 1,000 tokens
    # Savings: 90%
```

### 4.3 Routing & Fallback

```python
class SmartRouter:
    """Route queries to cheapest suitable model"""
    
    def __init__(self):
        self.models = {
            'simple': ('gpt-3.5-turbo', 0.001),
            'medium': ('gpt-4-turbo', 0.01),
            'complex': ('gpt-4-vision', 0.03),
        }
    
    def classify_query(self, query):
        """Determine task complexity"""
        
        # Heuristics
        if len(query) < 50 and "simple" in query.lower():
            return 'simple'
        elif any(w in query for w in ['analyze', 'reasoning', 'compare']):
            return 'complex'
        else:
            return 'medium'
    
    def route(self, query):
        """Pick cheapest model that can handle query"""
        
        complexity = self.classify_query(query)
        model, cost = self.models[complexity]
        return model
    
    def fallback_routing(self, query, first_model_failed=False):
        """Use cheaper model if possible; escalate if needed"""
        
        if not first_model_failed:
            return self.models['simple'][0]  # Try cheapest first
        else:
            return self.models['complex'][0]  # Escalate if simple fails

# Cost impact
# No routing: all queries use GPT-4 = $0.01 avg
# With routing: 50% simple ($0.0005) + 50% complex ($0.01) = $0.00525 avg
# Savings: 47.5%
```

---

## 5. Infrastructure Optimization

### 5.1 Caching Strategy

```python
class MultiLayerCache:
    """Cache at multiple levels"""
    
    def __init__(self):
        self.memory_cache = {}  # L1: In-memory (fast, limited)
        self.disk_cache = LRUCache('/cache')  # L2: Disk (slow, large)
        self.redis_cache = redis.Redis()  # L3: Shared (slow, shared)
    
    def get_or_compute(self, query, compute_fn):
        """Check all cache levels before computing"""
        
        # L1: Memory cache (100ms)
        if query in self.memory_cache:
            return self.memory_cache[query]
        
        # L2: Disk cache (500ms)
        if self.disk_cache.has(query):
            return self.disk_cache.get(query)
        
        # L3: Redis (50ms over network)
        if self.redis_cache.exists(query):
            return self.redis_cache.get(query)
        
        # Compute (5 seconds)
        result = compute_fn(query)
        
        # Store in all cache levels
        self.memory_cache[query] = result
        self.disk_cache.set(query, result)
        self.redis_cache.set(query, result, ttl=3600)
        
        return result

# Expected cache hit rates:
# First request: 0% (compute) = 5 seconds, $0.10
# Subsequent requests (same session): 90% (L1) = 0.1 sec, $0
# Requests from other users: 50-70% (L3) = 0.05 sec, $0
```

### 5.2 Batch Processing

```python
def batch_process_images(image_list, batch_size=100):
    """Process multiple images together to reduce overhead"""
    
    total_cost = 0
    results = []
    
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        
        # Single API call for batch
        batch_results = api.process_batch(batch)
        results.extend(batch_results)
        
        # API overhead per call: 0.0001 per image
        # Per-image processing: 0.001
        
        # Unbatched: 100 calls × (0.0001 + 0.001) = $0.11
        # Batched (1 call): 0.0001 + (100 × 0.001) = $0.101
        # Savings from batching: 8%
    
    return results

# Batch size trade-offs
# Size 10: Overhead = 1% per call
# Size 100: Overhead = 0.1% per call
# Size 1000: Overhead = 0.01% per call, but max latency
```

---

## Cost Optimization Checklist

- [ ] Image compression/resizing
- [ ] Video frame sampling (adaptive)
- [ ] Audio quality reduction
- [ ] Prompt optimization
- [ ] Model selection (right-size)
- [ ] Caching (multi-layer)
- [ ] Batch processing
- [ ] Concurrent requests
- [ ] Result reuse
- [ ] Fallback models

---

## Typical Cost Savings by Strategy

| Strategy | Implementation | Savings |
|----------|---|---|
| Image compression | JPEG quality reduction | 30-50% |
| Adaptive resolution | Task-specific sizing | 20-40% |
| Video sampling | Scene-aware frame extraction | 50-80% |
| Audio optimization | 16kHz + silence removal | 40-60% |
| Prompt optimization | Careful token budgeting | 30-50% |
| Model routing | Use cheaper models | 40-60% |
| Caching | Multi-layer cache | 60-85% |
| **Combined effect** | All strategies | **85-95%** |

---

## Key Takeaways

1. **Image**: Compress, resize, crop intelligently
2. **Audio**: Downsample, extract speech regions, choose model wisely
3. **Video**: Adaptive frame sampling is critical (biggest lever)
4. **LLM**: Route to cheapest suitable model, optimize prompts
5. **Infrastructure**: Cache aggressively, batch when possible
6. **Combined**: Can achieve 85-95% cost reduction

---

## Next Steps

1. Implement image caching (easiest, high impact)
2. Add adaptive video sampling (biggest lever)
3. Deploy smart model routing (consistent savings)
4. Monitor and optimize prompts (ongoing)

---

## References

- Cost-Efficient Fine-Tuning (Lester et al., 2021)
- Video Understanding Survey (Wang et al., 2021)
- Model Scaling Laws (Kaplan et al., 2020)
- LoRA: Low-Rank Adaptation (Hu et al., 2021)
