"""
Prefab Definitions
Defines standard configurations for creating common elements.
"""

# Prefab dictionary mapping prefab_name to its configuration.
# Configuration includes:
# - description: User-facing description of the prefab.
# - components: List of component configurations.
#   - type: The COMPONENT_TYPE name of the component.
#   - config: (Optional) Default configuration dictionary for the component.
# - required_configs: (Optional) Dictionary mapping component type names
#                     to a list of keys that MUST be provided in the 
#                     'config_overrides' when creating from this prefab.

PREFABS = {
    "simple_scratchpad": {
        "description": "A basic element for storing text notes.",
        "components": [
            {"type": "NoteStorageComponent"}, # State
            {"type": "ScratchpadActionHandler"}, # Logic/Tools
            {"type": "ToolProviderComponent"}, # Tool exposure
            {"type": "ScratchpadVeilProducer"} # VEIL representation
        ],
        "required_configs": {} # No special required configs for these components
    },
    
    # --- Add more prefabs here ---
    
    # Example: Placeholder for a message list (requires components to exist)
    # "discord_message_list": {
    #     "description": "Represents messages from a specific Discord channel.",
    #     "components": [
    #         {"type": "MessageListComponent", "config": {"channel_id": None}}, # Base config
    #         {"type": "ToolProviderComponent"},
    #         {"type": "MessageListVeilProducer"}
    #     ],
    #     "required_configs": {"MessageListComponent": ["channel_id"]} # Mark required overrides
    # },

    # Example: Uplink Proxy (Hypothetical - requires Uplink Components)
    # "uplink_proxy": {
    #     "description": "Connects to and caches state from a remote Space.",
    #     "components": [
    #         {"type": "UplinkConnectionComponent", "config": {"remote_space_id": None, "sync_interval": 60}},
    #         {"type": "RemoteStateCacheComponent", "config": {"remote_space_id": None, "cache_ttl": 300}},
    #         {"type": "ToolProviderComponent"},
    #         {"type": "UplinkVeilProducer"} 
    #         # Spaces inherit ContainerComponent and TimelineComponent
    #     ],
    #     # Requires remote_space_id to be provided via overrides for connection/cache
    #     "required_configs": {
    #           "UplinkConnectionComponent": ["remote_space_id"],
    #           "RemoteStateCacheComponent": ["remote_space_id"]
    #      }
    # }
} 