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
        "element_constructor_arg_keys": ["remote_space_id", "name", "description", "remote_space_info", "space_registry",],
        "components": [
            # UplinkProxy's __init__ adds its own core components:
            # UplinkConnectionComponent, RemoteStateCacheComponent, UplinkVeilProducer,
            # ToolProviderComponent (for local tools), and UplinkRemoteToolProviderComponent (for remote tools).
        ],
        "required_configs_for_element": ["remote_space_id", "name"] # 'description' is optional in constructor
    },

    "standard_chat_interface": {
        "description": "A standard chat interface element, typically used within SharedSpaces. Requires 'adapter_id' and 'external_conversation_id' in element_config to be set as attributes on the element.",
        "element_class_name": "BaseElement",
        "element_constructor_arg_keys": ["name", "description"],
        "components": [
            {"type": "ToolProviderComponent"},
            {
                "type": "MessageListComponent", 
                "config": {
                    "is_shared_channel_component": True 
                    # adapter_id and external_conversation_id will be set as element attributes
                    # and MessageListComponent can choose to read them from its owner.
                }
            },
            {
                "type": "MessageActionHandler", 
                "config": {
                    "is_for_shared_space_element": True
                    # adapter_id and external_conversation_id will be set as element attributes.
                    # MessageActionHandler can read these from its owner element for context.
                    # outgoing_action_callback would ideally be set by the system if the SharedSpace
                    # itself needs to send messages (e.g. system messages). For agent replies through
                    # an uplink, the Uplink's MAH will use the agent's callback.
                }
            },
            {"type": "MessageListVeilProducer"}
        ],
        "required_configs_for_element": ["name", "adapter_id", "external_conversation_id"],
        "element_attributes_from_config": {
            "adapter_id": "adapter_id",
            "external_conversation_id": "external_conversation_id",
            "adapter_type": "adapter_type",
            "server_name": "server_name",
            "conversation_name": "conversation_name"
        }
    },
} 