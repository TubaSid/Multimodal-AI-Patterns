# Vision-Language Chat: Production Example

A complete implementation of a vision-language chat system that combines image understanding with conversational AI. This example demonstrates late fusion architecture with streaming responses.

## Architecture

```
User Input (Image + Question)
    ↓
[Vision Encoder] → (768,)
    ↓
[Projection Layer] → (4096,)
    ↓
[Context Manager] → Conversation History + Visual Context
    ↓
[Language Model] → Streaming Token Generation
    ↓
User Output (Text Response)
```

## Key Features

- ✅ **Streaming responses** - Stream tokens as they're generated
- ✅ **Multi-turn conversation** - Maintain conversation history
- ✅ **Image caching** - Avoid reprocessing same images
- ✅ **Cost tracking** - Monitor API usage in real-time
- ✅ **Concurrent requests** - Handle multiple users
- ✅ **Error handling** - Graceful degradation

## Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:
```
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=claude-...
```

### 3. Run the Example

```bash
python main.py --image image.jpg --question "What's in this image?"
```

## Implementation Details

### Core Module: `vlm_system.py`

```python
class VisionLanguageChatSystem:
    def __init__(self, model='gpt-4-vision'):
        self.encoder = VisionEncoder()  # Frozen CLIP encoder
        self.projector = VisionProjector()  # Trainable projection
        self.language_model = LanguageModel(model)
        self.cache = ImageCache()
        
    def chat(self, image, question, history=[]):
        """Single turn of conversation"""
        
        # Encode image
        visual_features = self.cache.get_or_encode(
            image, 
            self.encoder
        )
        
        # Project to language space
        projected = self.projector(visual_features)
        
        # Build context
        context = self.build_context(projected, history)
        
        # Generate response (streaming)
        for token in self.language_model.generate_stream(
            context, 
            question
        ):
            yield token
```

## Files

- `main.py` - Entry point
- `vlm_system.py` - Core VLM system
- `encoders.py` - Vision and text encoders
- `cache.py` - Image/embedding cache
- `models.py` - Model wrappers
- `streaming.py` - Streaming utilities
- `requirements.txt` - Dependencies

## Performance Metrics

Expected performance on consumer hardware (RTX 3090):

- **First response latency**: 2-3 seconds
- **Subsequent responses** (cached): 0.5-1 second
- **Throughput**: 10-20 requests/second
- **Memory usage**: 8-12GB VRAM
- **Cost per request**: $0.02-0.05

## Next Steps

1. Try different language models
2. Fine-tune the projection layer
3. Add image preprocessing
4. Implement batch processing
5. Deploy as API service

## See Also

- [vision-language-models](../../skills/vision-language-models) - Detailed pattern guide
- [cost-optimization](../../skills/cost-optimization) - Reduce costs
- [embedding-spaces](../../skills/embedding-spaces) - Understand embeddings
