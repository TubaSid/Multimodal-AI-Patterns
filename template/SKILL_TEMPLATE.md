# Skill Template - Use This for New Contributions

Copy this template when creating new skills. Keep it under 600 lines for optimal performance.

```markdown
# Skill Name

## Overview

**One sentence:** What is this skill and why does it matter?

**Use this skill when:** List situations where this pattern applies.

---

## The Problem

What challenge or question does this skill address?
- Point 1
- Point 2
- Point 3

---

## Key Concepts

### Concept 1: Name
- Definition
- Why it matters
- Common misconception

### Concept 2: Name
- Definition
- Why it matters
- Common misconception

---

## Core Principles

1. **Principle Name**: Explanation
2. **Principle Name**: Explanation
3. **Principle Name**: Explanation

---

## Implementation

### Approach 1: [Simple/Basic]
```python
# Pseudocode or real code
class SimpleImplementation:
    def __init__(self):
        pass
    
    def process(self, input):
        # Implementation
        return output
```

**Pros:**
- Advantage 1
- Advantage 2

**Cons:**
- Limitation 1
- Limitation 2

**When to use:** Best for X situation

### Approach 2: [Advanced/Optimized]
```python
# More complex implementation
class AdvancedImplementation:
    pass
```

**Pros:**
- Advantage 1

**Cons:**
- Limitation 1

**When to use:** When X matters

### Approach 3: [Alternative]
```python
# Different strategy
class AlternativeImplementation:
    pass
```

**Pros:**
- Advantage 1

**Cons:**
- Limitation 1

**When to use:** When X is critical

---

## Comparison

| Aspect | Approach 1 | Approach 2 | Approach 3 |
|--------|-----------|-----------|-----------|
| Speed | [FAST][FAST][FAST] | [FAST][FAST] | [FAST] |
| Cost | $ | $$ | $$$ |
| Complexity | Simple | Medium | Complex |
| Best For | Situation A | Situation B | Situation C |

---

## Decision Tree

```
Are you optimizing for [metric]?
├─ YES → Use Approach 1
├─ NO → Is [constraint] important?
│  ├─ YES → Use Approach 2
│  └─ NO → Use Approach 3
└─ UNSURE → Start with Approach 1
```

---

## Common Mistakes

### [NO] Mistake 1: Description
**What goes wrong:** Explain the error

**Solution:** How to avoid it
```python
# DO THIS:
correct_code = ...

# NOT THIS:
wrong_code = ...
```

### [NO] Mistake 2: Description
**What goes wrong:** Explain the error

**Solution:** How to fix it

---

## Production Considerations

### Performance
- Typical latency: X-Y seconds
- Memory usage: X-Y GB
- Throughput: X-Y requests/second

### Cost
- API cost: $X per Y
- Infrastructure: $X per Z
- Total for typical workload: $X per month

### Scaling
- Handles X requests/second ✓
- Scales to Y concurrent users ✓
- Storage requirements: Z GB for typical use case

---

## Evaluation Metrics

How to measure if this pattern is working:

```python
# Metric 1: Metric Name
metric1 = calculate_metric1(results)
print(f"Metric 1: {metric1}")

# Metric 2: Metric Name
metric2 = calculate_metric2(results)
print(f"Metric 2: {metric2}")
```

**Good values:**
- Metric 1: > X (indicates working well)
- Metric 2: < Y (lower is better)

---

## Advanced Topics

### Topic 1: Deeper Dive
For advanced users, consider X technique when Y condition exists.

### Topic 2: Optimization
You can optimize further by doing X, gaining Y% improvement at Z cost.

### Topic 3: Customization
Adapt this pattern for X domain by modifying Y component.

---

## Real-World Example

### Scenario
**Context:** What real-world situation uses this?

**Challenge:** What problem did they face?

**Solution:** How they applied this pattern

**Results:** What improved? By how much?

```python
# Simplified example code showing real usage
```

---

## Troubleshooting

### Issue 1: [Problem Description]
**Symptom:** How does it manifest?

**Cause:** Why does this happen?

**Solution:**
```python
# Fix code here
```

### Issue 2: [Problem Description]
**Symptom:** How does it manifest?

**Cause:** Why does this happen?

**Solution:** Steps to fix

---

## Key Takeaways

1. **Principle 1:** Summary of key learning
2. **Principle 2:** Summary of key learning
3. **Principle 3:** Summary of key learning
4. **Principle 4:** Summary of key learning
5. **Principle 5:** Remember this

---

## Next Steps

1. **Easy:** Start with [Foundational Skill]
2. **Intermediate:** Learn [Related Skill]
3. **Advanced:** Explore [Advanced Skill]
4. **Build:** Try [Example Project]

---

## Resources

### Reading
- [Paper or Article 1](link) - Brief description
- [Paper or Article 2](link) - Brief description

### Tools
- [Tool 1](link) - What it does
- [Tool 2](link) - What it does

### References
- [Reference 1](link)
- [Reference 2](link)

---

## Related Skills

- [Vision-Language Models](../vision-language-models) - Related because...
- [Fusion Strategies](../fusion-strategies) - Related because...
- [Cost Optimization](../cost-optimization) - Related because...

---

## FAQ

**Q: Should I use this pattern for [situation]?**
A: It depends. Use it if X. Otherwise, try Y pattern.

**Q: How does this compare to [alternative]?**
A: Main differences:
- X: Better for situation A
- Y: Better for situation B

**Q: Can I combine this with [other pattern]?**
A: Yes! Best practices when combining:
1. Step 1
2. Step 2

---

## Contributing

Found an improvement or new variation? See [CONTRIBUTING.md](../../CONTRIBUTING.md) for how to contribute.

---

*Last Updated: [Date]*
*Status: Production-Ready*
*Version: 1.0*
```

---

## Tips for Great Skills

### Tone
- [YES] Conversational but precise
- [YES] Assume intermediate audience
- [YES] Explain jargon when introduced
- [NO] Don't be too casual or too academic

### Content
- [YES] Include real numbers (latency, cost, accuracy)
- [YES] Show code that actually works
- [YES] Explain why decisions matter
- [NO] Don't just list facts—explain reasoning

### Structure
- [YES] Build from simple to complex
- [YES] Use headers to organize
- [YES] Include code examples
- [YES] Add ASCII diagrams/tables
- [NO] Don't make it too long (< 600 lines)

### Accuracy
- [YES] Verify numbers and claims
- [YES] Test code snippets
- [YES] Reference sources
- [YES] Note when accuracy varies
- [NO] Don't guess or extrapolate

---

## Recommended Skills to Create

### High Priority (Most Requested)
- [ ] **Audio-Visual Fusion** - Synchronizing audio and video streams
- [ ] **Video Understanding** - Processing video for action/event recognition
- [ ] **Real-Time Streaming** - Low-latency multimodal processing
- [ ] **Multimodal RAG** - Combining text, image, audio in retrieval systems
- [ ] **Evaluation Frameworks** - Measuring multimodal system quality

### Medium Priority
- [ ] **Fine-Tuning Strategies** - Customizing pretrained models
- [ ] **Distillation & Compression** - Making models efficient
- [ ] **Multi-Agent Coordination** - Multiple specialists working together
- [ ] **Accessibility Patterns** - Video to audio, image descriptions, etc.

### Domain-Specific
- [ ] **Healthcare** - Medical imaging + clinical notes
- [ ] **Finance** - Document understanding + market data
- [ ] **E-Commerce** - Product images + descriptions + reviews
- [ ] **Content Creation** - Video + audio + text generation

---

**Start with a problem you know well. Write for someone like you from 6 months ago.**

Good luck!
