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

# Define a dictionary of element prefabs
# Each prefab specifies:
# - description: A human-readable description of what the prefab creates.
# - element_class_name: (Optional) The class name of the element to instantiate (defaults to BaseElement if not specified or if ElementFactory doesn't have specific logic).
# - element_constructor_arg_keys: (Optional) A list of keys from the `element_config` that should be passed as direct arguments to the element's constructor.
# - components: A list of component specifications. Each spec is a dict:
#   - "type": The string type of the component to add (must match COMPONENT_TYPE).
#   - "config": (Optional) A dictionary of configuration values for this component instance.
# - required_configs_for_element: (Optional) A list of keys that MUST be present in the `element_config` passed to the factory, primarily for setting up the element itself.
# - element_attributes_from_config: (Optional) A dictionary mapping keys from `element_config` to attribute names that will be set on the created element instance.
# - required_component_configs: (Optional) A dictionary where keys are component types (strings) and values are lists of required config keys for that specific component.

PREFABS = {
    "simple_scratchpad": {
        "description": "A basic element for storing text notes.",
        "element_constructor_arg_keys": ["name", "description"],
        "components": [
            {"type": "NoteStorageComponent"}, # State
            {"type": "ToolProviderComponent"}, # Tools
            {"type": "ScratchpadActionHandler"}, # Logic/Tools
            {"type": "ScratchpadVeilProducer"} # VEIL representation
        ],
        "required_configs_for_element": [] 
    },

    "standard_uplink_proxy": {
        "description": "Creates an UplinkProxy to connect to a remote Space. Requires 'remote_space_id' in element_config.",
        "element_class_name": "UplinkProxy", # Specifies the class to instantiate
        "element_constructor_arg_keys": ["remote_space_id", "name", "description", "remote_space_info"], # Added "remote_space_info"
        "components": [
            # UplinkProxy's __init__ adds its own core components:
            # UplinkConnectionComponent, RemoteStateCacheComponent, UplinkVeilProducerComponent, ToolProviderComponent.
            # MessageActionHandler is added here to provide tools for messaging via the uplink.
            {"type": "MessageActionHandler"}
        ],
        "required_configs_for_element": ["remote_space_id", "name"] # 'description' is optional in constructor
    },

    "direct_message_session": {
        "description": "A BaseElement configured to represent and handle a direct message session with a user on a specific adapter.",
        "element_class_name": "BaseElement", # Explicitly BaseElement
        "element_constructor_arg_keys": ["name", "description",],
        "components": [
            {"type": "ToolProviderComponent"},
            {"type": "MessageListComponent"},
            {"type": "MessageActionHandler"},
            {"type": "MessageListVeilProducer"},
        ],
        "required_configs_for_element": [ 
            "dm_adapter_id",
            "dm_external_conversation_id", 
            "dm_recipient_info" 
        ],
        "element_attributes_from_config": {
            "dm_adapter_id": "dm_adapter_id", 
            "dm_external_conversation_id": "dm_external_conversation_id", 
            "dm_recipient_info": "dm_recipient_info" 
        },
        "required_component_configs": {}
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