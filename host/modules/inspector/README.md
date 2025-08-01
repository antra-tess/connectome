# Connectome Inspector Module

The Inspector module provides a web-based interface for monitoring and inspecting the internal state of a running Connectome host.

## Features

- **Real-time System Status**: View overall system health, resource usage, and uptime
- **Space Inspection**: Detailed view of all spaces (Inner, Shared, Uplink) and their elements
- **Agent Monitoring**: Monitor agent configurations, components, and available tools
- **Adapter Status**: Check activity adapter connections and status
- **Performance Metrics**: System performance data including memory, CPU, and I/O statistics
- **Timeline DAG Inspection**: Browse and analyze Loom DAG timelines, events, and history across all spaces

## Configuration

Add these environment variables to your `.env` file:

```bash
# Enable the inspector web server
CONNECTOME_INSPECTOR_ENABLED=true

# Port for the inspector web server (default: 8080)
CONNECTOME_INSPECTOR_PORT=8080
```

## API Endpoints

Once enabled, the inspector provides these REST endpoints:

- `GET /` - API overview and available endpoints
- `GET /status` - Overall system status and health
- `GET /spaces` - Detailed information about all spaces and elements  
- `GET /agents` - Agent configurations and current status
- `GET /adapters` - Activity adapter connection status
- `GET /metrics` - System performance metrics
- `GET /timelines` - Timeline DAG overview for all spaces
- `GET /timelines/{space_id}` - Timeline details for specific space (supports ?limit=N query parameter)
- `GET /timelines/{space_id}/{timeline_id}` - Specific timeline events and details (supports ?limit=N query parameter)
- `GET /health` - Simple health check

## Usage

1. Enable the inspector in your `.env` file
2. Start the Connectome host: `python -m host.main`
3. Access the inspector at `http://localhost:8080`

## Example Responses

### System Status (`GET /status`)
```json
{
  "timestamp": 1704067200.0,
  "system": {
    "process_id": 12345,
    "uptime_seconds": 300.5
  },
  "memory": {
    "rss_mb": 128.5,
    "percent": 2.1
  },
  "spaces": {
    "total_spaces": 2,
    "inner_spaces": 1,
    "shared_spaces": 1
  }
}
```

### Spaces Data (`GET /spaces`)
```json
{
  "timestamp": 1704067200.0,
  "summary": {
    "total_spaces": 2,
    "inner_spaces": 1
  },
  "details": {
    "AdapterOfPlatform": {
      "id": "AdapterOfPlatform",
      "type": "InnerSpace",
      "elements": {...},
      "components": [...]
    }
  }
}
```

### Timeline Data (`GET /timelines`)
```json
{
  "timestamp": 1704067200.0,
  "summary": {
    "total_spaces_with_timelines": 2,
    "total_timelines": 2,
    "total_events": 45,
    "primary_timelines": 2
  },
  "spaces": {
    "agent_space_1": {
      "space_id": "agent_space_1",
      "space_type": "InnerSpace",
      "summary": {
        "has_timeline_component": true,
        "total_events": 23,
        "primary_timeline": "primary"
      },
      "timelines": {
        "primary": {
          "timeline_id": "primary",
          "is_primary": true,
          "head_event_ids": ["abc-123"],
          "recent_events_count": 10,
          "latest_event": {...},
          "event_types": {"message_received": 5, "tool_called": 3}
        }
      }
    }
  }
}
```

### Timeline Details (`GET /timelines/{space_id}/{timeline_id}`)
```json
{
  "timestamp": 1704067200.0,
  "space_id": "agent_space_1",
  "timeline_id": "primary",
  "timeline_info": {
    "is_primary": true,
    "head_event_ids": ["abc-123"],
    "total_events": 23
  },
  "events": [
    {
      "id": "abc-123",
      "timestamp": 1704067150.0,
      "parent_ids": ["def-456"],
      "timeline_id": "primary",
      "payload": {
        "event_type": "message_received",
        "data": {...}
      }
    }
  ],
  "events_returned": 10,
  "events_limited": false
}
```

## Security Considerations

- The inspector server binds to `localhost` only by default
- No authentication is currently implemented - suitable for development/debugging only
- Auth tokens in adapter configurations are masked in responses
- The inspector provides read-only access to system state

## Development

The inspector consists of:

- `inspector_server.py` - HTTP server and route handlers
- `data_collector.py` - System state collection logic
- Integration with the main host process in `host/main.py`

To extend the inspector:

1. Add new collection methods to `InspectorDataCollector`
2. Add corresponding route handlers to `InspectorServer`
3. Update the API documentation above