# Consyn AI API Reference

## Overview

The Consyn AI API provides a robust and flexible interface for text generation across multiple model variants.

## Base URL
`https://api.consyn.ai/v1`

## Authentication
API key required for all requests. Include in header:
```
Authorization: Bearer YOUR_API_KEY
```

## Available Endpoints

### 1. Generate Text
`POST /generate`

#### Request Parameters
- `prompt` (string, required): Input text prompt
- `model_name` (string, default: "verse"): 
  - Allowed values: "verse", "stanza", "epic"
- `max_new_tokens` (integer, default: 128): Maximum tokens to generate
- `temperature` (float, default: 0.7): Sampling temperature
- `top_k` (integer, default: 50): Top-k filtering
- `top_p` (float, default: 0.9): Nucleus sampling
- `do_sample` (boolean, default: true): Enable sampling

#### Example Request
```json
{
    "prompt": "Once upon a time",
    "model_name": "stanza",
    "max_new_tokens": 256,
    "temperature": 0.8
}
```

#### Response
```json
{
    "generated_text": "Once upon a time, in a world where technology...",
    "generation_time": 0.342
}
```

### 2. Batch Generation
`POST /generate/batch`

#### Request Parameters
- `prompts` (array of strings): List of input prompts
- `model_name` (string, default: "verse"): Model to use
- `batch_size` (integer, default: 4): Processing batch size
- `max_new_tokens` (integer, default: 128): Maximum tokens per generation
- Other generation parameters similar to single generation

#### Example Request
```json
{
    "prompts": [
        "In the future,", 
        "Once upon a time,", 
        "The science of"
    ],
    "model_name": "epic",
    "max_new_tokens": 192
}
```

#### Response
```json
{
    "generated_texts": [
        "In the future, artificial intelligence...",
        "Once upon a time, in a magical kingdom...",
        "The science of quantum mechanics revolutionized..."
    ],
    "generation_time": 0.576
}
```

### 3. Streaming Generation
`POST /generate/stream`

#### Request Parameters
- Identical to single generation endpoint
- Supports streaming response via Server-Sent Events (SSE)

#### Response
Streaming JSON objects:
```json
{"token": "Once", "index": 1, "finish_reason": null}
{"token": " upon", "index": 2, "finish_reason": null}
...
{"token": "", "index": 128, "finish_reason": "stop"}
```

### 4. Model Information
`GET /models`

#### Response
```json
[
    {
        "model_name": "verse",
        "model_type": "Consyn Verse",
        "parameters": 125000000,
        "max_context_length": 1024,
        "vocab_size": 50257,
        "loaded": true
    },
    ...
]
```

### 5. Health Check
`GET /health`

#### Response
```json
{
    "status": "healthy",
    "models_loaded": true,
    "gpu_available": true,
    "gpu_count": 1
}
```

## Error Handling

### Common Error Codes
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Authentication failed
- `403 Forbidden`: Insufficient permissions
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side issues

### Error Response Format
```json
{
    "error": "Invalid model name",
    "details": "Model must be one of: verse, stanza, epic",
    "request_id": "abc123"
}
```

## Rate Limiting

- Default: 100 requests/minute per API key
- Burst limit: 10 concurrent requests
- Exceeded limits return 429 status

## Best Practices

1. Use appropriate model for your task
2. Implement retry mechanisms
3. Handle streaming responses
4. Set reasonable token limits
5. Use temperature for creativity control

## Python SDK Example

```python
from consyn import ConsynClient

client = ConsynClient(api_key='your_api_key')

# Single generation
response = client.generate(
    prompt="Tell me a story about",
    model='stanza',
    max_tokens=256
)

# Batch generation
responses = client.batch_generate([
    "In Paris,", 
    "During the Renaissance,"
])

# Streaming generation
for token in client.generate_stream("Once upon a time"):
    print(token, end='')
```

## Pricing

### Free Tier
- 5,000 tokens/month
- Verse model only
- Limited rate limits

### Professional Tiers
- Pay-as-you-go
- Multiple model access
- Higher rate limits
- Priority support

### Pricing Model
- Verse: $0.0001 per 1000 tokens
- Stanza: $0.001 per 1000 tokens
- Epic: $0.01 per 1000 tokens

## Compliance and Security

- GDPR compliant
- SOC 2 Type II certified
- End-to-end encryption
- No data retention
- Prompt filtering

## Support

- Documentation: [docs.consyn.ai](https://docs.consyn.ai)
- Email: support@consyn.ai
- Discord Community
- GitHub Discussions

## Changelog

### v1.0.0 (Current)
- Initial API release
- Three model variants
- Streaming support
- Batch generation

### Upcoming Features
- More model sizes
- Multi-language support
- Enhanced few-shot capabilities
- Fine-tuning endpoints