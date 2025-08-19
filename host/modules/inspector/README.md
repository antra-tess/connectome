# Connectome Inspector Module

The Inspector module provides multiple interfaces for monitoring and inspecting the internal state of a running Connectome host, including web APIs, CLI tools, and a powerful interactive TUI.

## Interfaces

### IPC-TUI (Interactive Terminal UI)
The most powerful inspection tool - a real-time terminal interface with tree navigation, editing capabilities, and cursor-based pagination.

```bash
# Launch IPC-TUI interface (recommended)
./tui_inspector

# Alternative launch method
python -m host.modules.inspector.ipc_tui
```

**Navigation:**
- **Arrow keys**: Navigate through tree nodes
- **Enter**: Expand/collapse nodes or drill into details  
- **Backspace**: Go back to parent view
- **PgUp/PgDn**: Scroll through long lists with pagination
- **Home/End**: Jump to start/end of current view
- **'q'**: Quit application

### Web API
RESTful JSON endpoints for programmatic access and web interfaces.

### CLI Tools
Command-line utilities for quick inspection and debugging.

## Features

- **Real-time System Status**: View overall system health, resource usage, and uptime
- **Space Inspection**: Detailed view of all spaces (Inner, Shared, Uplink) and their elements
- **Agent Monitoring**: Monitor agent configurations, components, and available tools
- **Adapter Status**: Check activity adapter connections and status
- **Performance Metrics**: System performance data including memory, CPU, and I/O statistics
- **Timeline DAG Inspection**: Browse and analyze Loom DAG timelines, events, and history across all spaces
- **VEIL Facet Browsing**: Navigate VEIL operations and state updates with pagination
- **Interactive Editing**: Modify configurations, component states, and system parameters directly
- **Write Operations**: Full duplex system control - read and modify live system state
- **Future REPL Integration**: Planned Python REPL and debugger integration for advanced system interaction

## Configuration

Add these environment variables to your `.env` file:

```bash
# Enable the inspector web server
CONNECTOME_INSPECTOR_ENABLED=true

# Port for the inspector web server (default: 8080)
CONNECTOME_INSPECTOR_PORT=8080
```

## Pagination System

Inspector uses **cursor-based pagination** optimized for real-time data streams where new entries appear at the top.

### Timeline Events
- **Cursor**: timestamp (float)
- **Positive limit**: newer events (forward chronological)  
- **Negative limit**: older events (reverse chronological)

### VEIL Facets
- **Cursor**: facetID (string)
- **Positive limit**: forward through facet list
- **Negative limit**: backward through facet list

## API Endpoints

### Web API (HTTP)

When Inspector server is running (default port 8080):

- `GET /` - API overview and available endpoints
- `GET /status` - Overall system status and health
- `GET /spaces` - Detailed information about all spaces and elements  
- `GET /spaces/{space_id}/veil-facets?cursor={facetID}&limit={±N}` - VEIL facets (paginated)
- `GET /spaces/{space_id}/timelines/{timeline_id}/events?offset={timestamp}&limit={±N}` - Timeline events (paginated)
- `GET /agents` - Agent configurations and current status
- `GET /adapters` - Activity adapter connection status
- `GET /metrics` - System performance metrics
- `GET /timelines` - Timeline DAG overview for all spaces
- `GET /timelines/{space_id}` - Timeline details for specific space (supports ?limit=N query parameter)
- `GET /timelines/{space_id}/{timeline_id}` - Specific timeline events and details (supports ?limit=N query parameter)
- `GET /health` - Simple health check

### IPC Commands

For programmatic access via `IPCClient`:

- **`status`** - System status
- **`spaces`** - Spaces listing  
- **`veil-facets`** - Paginated facets for space
- **`timeline-details`** - Paginated timeline events
- **`agent-details`** - Agent configuration details

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
- The inspector provides **full read-write access** to system state - use with caution in production

## Common Issues & Debugging

Based on extensive debugging sessions, here are the most frequent problems when working with Inspector/IPC-TUI:

### 1. Pagination Logic Errors
**Symptoms**: Timeline shows newer events when requesting older ones  
**Location**: `data_collector.py` filtering logic, `ipc_tui.py:2340-2378`  
**Fix**: Negative limit should filter `timestamp < offset` for older events  
**Debug**: Check `_initialize_pagination_state()` cursor/limit handling  
**Test**: Navigate to timeline details, scroll down - should load older events

### 2. Facet List Duplication
**Symptoms**: Same facets appearing multiple times during scroll  
**Location**: `data_collector.py:1100` cursor management  
**Fix**: Ensure `next_cursor` advances correctly and deduplication works  
**Debug**: Monitor `limited_facets[-1].facet_id` cursor assignment  
**Test**: Navigate to VEIL facets, scroll - no repeats should appear

### 3. Scroll Position Lost
**Symptoms**: Cursor jumps around during expand/collapse operations  
**Location**: Tree navigation state management  
**Fix**: Properly track `_find_current_node_index()` and restore position  
**Debug**: Check `_check_pagination_on_scroll()` direction handling  
**Test**: Expand/collapse nodes while scrolling - position should maintain

### 4. Write Operation Issues
**Symptoms**: External editor not launching, changes not saving, or write endpoints failing  
**Location**: Write operation handling across all Inspector interfaces  
**Fix**: Verify `write_endpoint` configuration, temp file handling, and endpoint permissions  
**Debug**: Check `handle_edit_action()` subprocess calls and write endpoint responses  
**Test**: Try editing space configurations, component settings, or system parameters

### 5. Navigation State Loss
**Symptoms**: Back button collapses entire tree unexpectedly  
**Location**: Navigation mode transitions  
**Fix**: Maintain tree expansion state in navigation stack  
**Debug**: Monitor `NavigationMode` transitions and node `is_expanded` flags  
**Test**: Navigate deep into tree, use back button - expansion should restore

### Debugging Commands

```bash
# Enable debug logging
CONNECTOME_LOG_LEVEL=DEBUG python -m host.main

# Run IPC-TUI with logging  
CONNECTOME_LOG_LEVEL=DEBUG python -m host.modules.inspector.ipc_tui

# Filter for pagination issues
CONNECTOME_LOG_LEVEL=DEBUG python -m host.main 2>&1 | grep -i "pagination\|cursor\|facet"
```

## Development

### Architecture

```
Connectome Host → InspectorDataCollector → EndpointHandlers → {Web API, IPC-TUI, CLI}
```

**Core Files:**
- `inspector_server.py` - Web-based JSON API server
- `data_collector.py` - Core data gathering and formatting logic  
- `ipc_server.py` / `ipc_client.py` - Inter-process communication layer
- `ipc_tui.py` - Terminal-based interactive interface (IPC-TUI)
- `endpoint_handlers.py` - Shared endpoint logic across interfaces
- `cli_inspector.py` / `cli_handler.py` - Command-line tools

### Adding New Endpoints

1. **Implement data collection** in `InspectorDataCollector`
2. **Add endpoint handler** in `InspectorEndpointHandlers`  
3. **Register web route** in `InspectorServer` (if needed)
4. **Add IPC command** in `IPCServer` (if needed)
5. **Update TUI navigation** in `IPCTUIInspector` (if needed)

### Adding Pagination Support

1. **Implement cursor logic** in `data_collector.py` - ensure proper filtering direction
2. **Add pagination state** in `ipc_tui.py` - cursor initialization and updates  
3. **Handle edge cases** - first page, last page, empty results
4. **Test both directions** - positive and negative limits
5. **Verify deduplication** - ensure no repeated entries across pages

### Testing Checklist

When modifying pagination or navigation:

- [ ] Timeline events load older entries when scrolling down
- [ ] VEIL facets don't duplicate during scroll  
- [ ] Tree expansion state preserved during navigation
- [ ] Cursor position maintained during node collapse/expand
- [ ] Write operations work correctly across all interfaces
- [ ] Editable fields can be modified and saved
- [ ] Edge cases handled (empty results, single page)
- [ ] Both positive and negative limits work correctly

### Contributing

When working on Inspector module:

1. **Test extensively** - the TUI has many edge cases
2. **Check all pagination directions** - positive and negative limits  
3. **Verify state persistence** - navigation, expansion, cursor position
4. **Test with real data** - empty states, large datasets, rapid updates
5. **Update documentation** - both this README and API docs

### TODOs and Known issues:

- in TUI, Agents page, agents/{agent} path is detected as drill-downable. However, it does not drill down to anything useful, in fact it errors out - "Unknown command: Agent details". Instead, somewhere there should be a way to access the *rendered* HUD state of the agent; this will require a new endpoint.