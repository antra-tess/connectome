# Chat Log to DAG Converter

This utility converts arbitrary chat logs into Connectome-compatible DAG history files that can be loaded and replayed by the system.

## Overview

The `chat_log_to_dag.py` utility:

- **Reuses existing Connectome code** - Uses `TimelineComponent` and storage patterns
- **Supports multiple input formats** - JSON and CSV chat logs
- **Creates proper DAG structure** - Timeline events with parent-child relationships  
- **Generates storage-compatible output** - Files that match Connectome's persistence format
- **Preserves message ordering** - Maintains chronological timeline structure

## Usage

### Basic Usage

```bash
# Convert JSON chat log
python -m tools.chat_log_to_dag --input chat.json --output timeline.json --space-id demo_space

# Convert CSV chat log  
python -m tools.chat_log_to_dag --input chat.csv --format csv --output timeline.json --space-id demo_space

# With custom adapter ID and debug logging
python -m tools.chat_log_to_dag --input chat.json --output timeline.json --space-id demo_space --adapter-id discord_import --log-level DEBUG
```

### Command Line Options

- `--input, -i`: Input chat log file (JSON or CSV) [required]
- `--output, -o`: Output DAG file path (JSON) [required] 
- `--space-id`: Space ID for the DAG [required]
- `--format`: Input format (`json`, `csv`, or `auto`) [default: `auto`]
- `--adapter-id`: Adapter ID to use in events [default: `chat_log_importer`]
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) [default: `INFO`]

## Input Formats

### Minimal Requirements

**Only `text` is required** - all other fields are optional and will be auto-generated:

- **Auto-generated timestamps** - Realistic conversation timing patterns
- **Auto-generated sender IDs** - Content-based hashing for consistency  
- **Auto-generated sender names** - Deterministic name selection
- **Auto-generated message IDs** - Timestamp + content hash

### JSON Format

**Ultra-minimal (text-only array):**
```json
["Hello world!", "How are you?", "I'm doing great!"]
```

**Minimal (object with text only):**
```json
[
  {"text": "Hello world!"},
  {"text": "How are you?"},
  {"text": "I'm doing great!"} 
]
```

**Full format with optional fields:**
```json
{
  "messages": [
    {
      "text": "Hello everyone!",
      "sender_id": "user123",
      "sender_name": "Alice Smith", 
      "timestamp": "2024-01-15T10:30:00Z",
      "message_id": "msg_001",
      "conversation_id": "general_chat",
      "is_dm": false,
      "mentions": [],
      "attachments": []
    }
  ]
}
```

**Supported field mappings:**
- `text`, `content`, `message` → message text *(required)*
- `sender_id`, `user_id`, `author` → sender identifier  
- `sender_name`, `username`, `display_name` → sender display name
- `timestamp`, `time`, `created_at` → message timestamp
- `message_id`, `id` → message identifier
- `is_dm`, `direct_message` → direct message flag
- `conversation_id`, `channel_id`, `chat_id` → conversation identifier
- `mentions` → list of mentioned users
- `attachments` → list of file attachments

### CSV Format

**Minimal CSV (text only):**
```csv
text
"Hello world!"
"How are you?"
"I'm doing great!"
```

**Full CSV with optional fields:**
```csv
text,sender_id,sender_name,timestamp,conversation_id,is_dm,mentions
"Hello everyone!",user123,"Alice Smith",2024-01-15T10:30:00Z,general_chat,false,[]
```

Maps CSV columns to message fields using the same field names as JSON. JSON-formatted strings are supported for complex fields like `mentions` and `attachments`.

## Intelligent Defaults (Procgen Heuristics)

When fields are missing, the tool generates meaningful defaults:

### Timestamp Generation
- **First message**: January 1, 2024, 9:00 AM UTC (realistic start time)
- **Conversation patterns**: Realistic timing with bursts and pauses
  - Within bursts: 30 seconds to 3 minutes between messages
  - Between bursts: 5 minutes to 2 hours gaps
  - Natural randomness added for realism

### Sender ID Generation  
- **Content-based hashing**: Uses first 3 words of message text
- **Consistent mapping**: Same message content → same sender ID
- **Format**: `user_abc12345` (8-character hash)

### Sender Name Generation
- **Deterministic selection**: Hash-based name picking
- **Realistic names**: 16 first names × 14 last names = 224 combinations
- **Examples**: "Alice Smith", "Bob Johnson", "Maya Martin"

### Message ID Generation
- **Format**: `msg_timestamp_hash` 
- **Components**: Last 6 digits of timestamp + 6-character content hash
- **Example**: `msg_099600_420e57`

### Conversation ID
- **Default**: `imported_conversation`
- **Consistent grouping**: All messages in same conversation

## Output Format

The utility generates three files:

1. **Timeline State** (`timeline_state_{space_id}.json`) - Timeline metadata
2. **Timeline Events** (`timeline_events_{space_id}.json`) - Event data  
3. **Combined Format** (`{output_path}`) - Inspection-friendly combined file

### Timeline State Structure
```json
{
  "state_key": "timeline_state_demo_space",
  "data": {
    "_timelines": {
      "primary": {
        "is_primary": true,
        "head_event_ids": ["event_uuid"]
      }
    },
    "_primary_timeline_id": "primary",
    "last_updated": 1705320000.0
  },
  "stored_at": "2024-01-15T12:00:00Z"
}
```

### Timeline Events Structure
```json
{
  "state_key": "timeline_events_demo_space", 
  "data": {
    "_all_events": {
      "event_uuid": {
        "id": "event_uuid",
        "timestamp": 1705317000.0,
        "parent_ids": ["parent_event_uuid"],
        "timeline_id": "primary",
        "payload": {
          "event_type": "message_received",
          "payload": {
            "source_adapter_id": "chat_log_importer",
            "text": "Hello everyone!",
            "sender_external_id": "user123",
            "sender_display_name": "Alice Smith",
            "timestamp": 1705317000.0,
            "is_dm": false,
            "mentions": [],
            "attachments": []
            // ... additional fields
          }
        }
      }
    },
    "last_updated": 1705320000.0
  },
  "stored_at": "2024-01-15T12:00:00Z"  
}
```

## Integration with Connectome

### Loading Generated DAG Files

To load the generated DAG files into a running Connectome system:

1. **Place files in storage directory:**
   ```bash
   # Copy to system storage directory
   cp timeline_state_demo_space.json storage_data/system/
   cp timeline_events_demo_space.json storage_data/system/
   ```

2. **Configure agent for the space:**
   ```bash
   export CONNECTOME_AGENTS_JSON='[{
     "agent_id": "demo_agent",
     "name": "Demo Agent", 
     "space_id": "demo_space",
     "agent_loop_component_type_name": "SimpleRequestResponseLoopComponent"
   }]'
   ```

3. **Start with event replay enabled:**
   ```bash
   export CONNECTOME_EVENT_REPLAY_MODE=enabled
   python -m host.main
   ```

### Event Replay Process

When starting with replay enabled:

1. `TimelineComponent` loads existing DAG structure from storage
2. Events are replayed chronologically to reconstruct state
3. `MessageListComponent` rebuilds message history
4. Agent memory and context are restored
5. System continues from the last DAG state

## Architecture Details

### Code Reuse

The utility leverages existing Connectome components:

- **`TimelineComponent`** - DAG event management and persistence logic
- **`BaseElement`** - Component container architecture  
- **`FileStorage`** - Storage format and serialization patterns
- **Event payload structures** - Compatible with `ExternalEventRouter` format

### Message Normalization

The `ChatMessage` class normalizes various input formats:

- **Timestamp handling** - Supports Unix timestamps, ISO strings, datetime objects
- **Field mapping** - Maps common field names across different chat platforms
- **Metadata preservation** - Stores additional fields as metadata
- **Connectome payload generation** - Creates proper event payloads

### DAG Structure Generation

Messages are converted to a linear DAG:

1. **Root event** - Timeline creation event
2. **Sequential events** - Each message becomes a timeline event
3. **Parent-child links** - Events link to previous event (linear chain)
4. **Proper ordering** - Messages sorted by timestamp before processing

## Examples

See the `examples/` directory for sample input files:

- `example_chat.json` - JSON format example
- `example_chat.csv` - CSV format example

Run the examples:

```bash
# JSON example
python -m tools.chat_log_to_dag --input tools/examples/example_chat.json --output example_output.json --space-id example_space

# CSV example  
python -m tools.chat_log_to_dag --input tools/examples/example_chat.csv --output example_output.json --space-id example_space
```

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure you're running from the Connectome root directory
- Install required dependencies: `pip install -r requirements.txt`

**Timestamp parsing errors:**
- Check timestamp format in input logs
- Supported formats: Unix timestamps, ISO 8601 strings, datetime objects

**Empty output:**
- Verify input file contains messages
- Check field name mappings for your chat platform
- Use `--log-level DEBUG` for detailed parsing information

**Storage format issues:**
- Generated files should be compatible with both `FileStorage` and `SQLiteStorage`
- Verify JSON structure matches expected storage format

### Debugging

Enable debug logging to see detailed processing:

```bash
python -m tools.chat_log_to_dag --input chat.json --output timeline.json --space-id demo_space --log-level DEBUG
```

This shows:
- Message parsing details
- Event creation process  
- DAG structure generation
- File output operations