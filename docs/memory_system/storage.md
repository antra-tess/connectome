# Storage Architecture

## Overview
The system uses a file-based storage approach optimized for debugging, performance, and maintainability. While this introduces some data denormalization, the benefits of improved inspectability and operational efficiency outweigh the costs.

## Directory Structure
```
conversations/
    {conversation_id}/
        current.json         # Typingcloud format with compressed memories
        raw_messages.json    # Complete history of uncompressed messages
        chunks.json         # Record of chunking decisions
        memories.json       # Record of memory formation
cache/
    {chunk_hash}.json       # LLM generation cache
```

## File Descriptions

### current.json
- Maintains Typingcloud format compatibility
- Contains compressed memories as properly tagged messages
- Includes system messages marking memory sections
- Preserves all original Typingcloud fields

### raw_messages.json
- Stores all original messages chronologically
- Never gets compressed or modified
- Essential for:
  * Recompressing with new context
  * Accurate tail calculations
  * Preserving original conversation flow

### chunks.json
- Records how messages were grouped into chunks
- Stores:
  * Chunk boundaries
  * Token counts
  * Hash values
  * References to contained messages
  * Link to any memory created from this chunk

### memories.json
- Documents the memory formation process
- Contains:
  * Source chunk references
  * Complete 6-message memory formation sequence
  * Timestamps and model information
  * Compression iteration tracking

### Cache Files
- Named by chunk hash
- Contains:
  * LLM generations for quote requests
  * Perspective prompts
  * Analysis summaries
  * Context manager interactions

## Benefits of Denormalization

1. Debugging and Verification
- Easy inspection of chunking decisions
- Clear view of memory formation process
- Traceable compression history

2. Performance
- Avoid recomputing chunks
- Quick access to memory structures
- Efficient retrieval of related data

3. Maintenance
- Better visibility into system state
- Easier to identify issues
- Simplified debugging process

4. Recompression Support
- Clear tracking of compression history
- Easy identification of recompression needs
- Maintained chunk-memory relationships

## Trade-offs

### Advantages
- Improved system inspectability
- Better debugging capabilities
- Faster access to derived data
- Clearer compression history

### Disadvantages
- Data redundancy
- More storage space required
- Need to maintain consistency across files
- More complex update procedures

The benefits of improved debugging, maintenance, and operational efficiency are deemed to outweigh the costs of data denormalization.
