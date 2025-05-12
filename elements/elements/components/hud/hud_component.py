import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from ..base_component import Component
# Needs access to the SpaceVeilProducer on the owner (InnerSpace)
from ..space.space_veil_producer import SpaceVeilProducer
# Import the registry decorator
from elements.component_registry import register_component
# May need access to GlobalAttentionComponent for filtering
# from ..attention.global_attention_component import GlobalAttentionComponent

if TYPE_CHECKING:
    # May need LLM provider for summarization/rendering assistance
    from llm.provider_interface import LLMProviderInterface

logger = logging.getLogger(__name__)

@register_component
class HUDComponent(Component):
    """
    Head-Up Display Component.

    Responsible for generating a contextual representation (e.g., a prompt)
    of the agent's current state based on the InnerSpace's aggregated VEIL.
    """
    COMPONENT_TYPE = "HUDComponent"

    # Dependencies that InnerSpace should inject
    # Optional: LLMProvider for advanced rendering/summarization
    INJECTED_DEPENDENCIES = {
        'llm_provider': '_llm_provider'
    }

    def __init__(self, element=None, llm_provider: Optional['LLMProviderInterface'] = None, **kwargs):
        super().__init__(element, **kwargs)
        self._llm_provider = llm_provider # Optional LLM for advanced processing

    def initialize(self, **kwargs) -> None:
        """Initializes the HUD component."""
        super().initialize(**kwargs)
        logger.debug(f"HUDComponent initialized for Element {self.owner.id}")
        if self._llm_provider:
            logger.debug(f"HUDComponent using LLM provider: {self._llm_provider.__class__.__name__}")

    def _get_space_veil_producer(self) -> Optional[SpaceVeilProducer]:
        """Helper to get the SpaceVeilProducer from the owning InnerSpace."""
        if not self.owner:
            return None
        # Assuming SpaceVeilProducer is the primary/only VEIL producer on InnerSpace
        return self.owner.get_component(SpaceVeilProducer)

    # def _get_global_attention(self) -> Optional[GlobalAttentionComponent]:
    #     """Helper to get the GlobalAttentionComponent from the owning InnerSpace."""
    #     if not self.owner:
    #         return None
    #     return self.owner.get_component(GlobalAttentionComponent)

    async def get_agent_context(self, options: Optional[Dict[str, Any]] = None) -> str:
        """
        (Async) Generates the agent's context, suitable for an LLM prompt.

        Retrieves the aggregated VEIL from the InnerSpace, filters/renders it,
        and returns a structured string representation.

        Args:
            options: Optional dictionary controlling context generation
                     (e.g., verbosity, max_tokens, focus_element_id,
                      render_style: 'clean' (default) or 'verbose_tags').

        Returns:
            A string representing the agent's current context.
        """
        logger.debug(f"Generating agent context for {self.owner.id}...")
        options = options or {}

        # 1. Get the full aggregated VEIL from InnerSpace's producer
        veil_producer = self._get_space_veil_producer()
        if not veil_producer:
            logger.error(f"[{self.owner.id}] Cannot generate context: SpaceVeilProducer not found.")
            return "Error: Could not retrieve internal state."

        try:
            full_veil = veil_producer.get_full_veil()
            if not full_veil:
                 logger.warning(f"[{self.owner.id}] SpaceVeilProducer returned empty VEIL.")
                 return "Current context is empty."
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error getting full VEIL from SpaceVeilProducer: {e}", exc_info=True)
            return f"Error: Failed to retrieve internal state - {e}"

        # 2. Get attention signals (optional filtering)
        # attention_comp = self._get_global_attention()
        attention_requests = {} # attention_comp.get_attention_requests() if attention_comp else {}

        # 3. Render the VEIL structure into a string
        #    This is the core logic block. Needs a recursive helper function.
        #    It should traverse the 'full_veil' dict, paying attention to node_type,
        #    properties, and children. It should format specific node types
        #    (like message_list_container, message_item, uplink_proxy_root)
        #    in a readable way. It might use attention_requests to prioritize or
        #    highlight certain parts.
        try:
            # Render synchronously
            context_string = self._render_veil_node_to_string(full_veil, attention_requests, options, indent=0)
        except Exception as e:
            logger.error(f"[{self.owner.id}] Error rendering VEIL to string: {e}", exc_info=True)
            # Maybe return partial context or just the error?
            # Let's try to return the raw VEIL structure on render error for debugging
            import json
            return f"Error rendering context: {e}\nRaw VEIL:\n{json.dumps(full_veil, indent=2)}"


        # 4. Optional Post-processing (e.g., summarization with LLM if needed/available)
        # if self._llm_provider and options.get("summarize"):
        #     context_string = self._summarize_context(context_string)

        logger.info(f"Generated agent context for {self.owner.id} (approx length: {len(context_string)}).")
        return context_string

    # --- Rendering Helpers --- 

    def _render_default(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Default fallback renderer - shows type and dumps properties."""
        props = node.get("properties", {})
        output = f"{indent_str}>> Default Render for {node_info}:\n"
        for key, value in props.items():
            # Skip annotations we already used for dispatch/display
            if key in ["structural_role", "content_nature", "rendering_hint"]:
                 continue
            # Avoid rendering huge child lists embedded in properties
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                 output += f"{indent_str}  {key}: [complex data omitted]\n"
            else:
                 output += f"{indent_str}  {key}: {value}\n"
        return output

    def _render_container(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a container node - often just acts as a header for children."""
        props = node.get("properties", {})
        # Display name if available, otherwise use type
        name = props.get('element_name', node.get('node_type'))
        output = f"{indent_str}{name}:\n" # Simple header
        # Children are handled by the main loop
        return output

    def _render_chat_message(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a chat message node cleanly."""
        props = node.get("properties", {})
        sender = props.get("sender_name", "Unknown")
        text = props.get("text_content", "")
        timestamp = props.get("timestamp", "") # Prefer raw timestamp if available for sorting/display
        # TODO: Convert timestamp to human-readable format based on options?
        is_edited = props.get("is_edited", False)

        # Simple chat format
        output = f"{indent_str}{sender}: {text}"
        if is_edited:
            output += " (edited)"
        # output += f" [@ {timestamp}]" # Optional timestamp
        output += "\n"
        return output

    def _render_uplink_summary(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent_str: str, node_info: str) -> str:
        """Renders a cleaner summary for an uplink node."""
        props = node.get("properties", {})
        remote_name = props.get("remote_space_name", "Unknown Space")
        remote_id = props.get("remote_space_id", "unknown_id")
        node_count = props.get("cached_node_count", 0)
        # Add specific title for uplink section
        output = f"{indent_str}Uplink to {remote_name} ({remote_id}):\n"
        output += f"{indent_str}  (Contains {node_count} cached items)\n"
        # Children (the cached items) will be rendered by the main loop
        return output

    # --- Main Recursive Renderer (Dispatcher) --- 

    def _render_veil_node_to_string(self, node: Dict[str, Any], attention: Dict[str, Any], options: Dict[str, Any], indent: int = 0) -> str:
        """
        Recursively renders a VEIL node and its children into a string.
        Dispatches rendering based on node properties/annotations.
        Supports different rendering styles via options['render_style'].
        """
        indent_str = "  " * indent
        node_type = node.get("node_type", "unknown")
        props = node.get("properties", {})
        children = node.get("children", [])
        node_id = node.get("veil_id")

        # --- Determine Rendering Strategy --- 
        structural_role = props.get("structural_role")
        content_nature = props.get("content_nature")
        rendering_hint = props.get("rendering_hint")

        # Determine rendering style
        render_style = options.get("render_style", "clean") # Default to clean
        use_verbose_tags = (render_style == "verbose_tags")

        # Basic info string for logging/default rendering
        node_info = f"Type='{node_type}', Role='{structural_role}', Nature='{content_nature}', ID='{node_id}'"

        # Decide which renderer to use based on hints/type (Order matters)
        render_func = self._render_default # Default fallback
        
        if content_nature == "chat_message":
            render_func = self._render_chat_message
        elif content_nature == "uplink_summary":
             render_func = self._render_uplink_summary
        elif structural_role == "container" or node_type == "message_list_container": # Treat message list container like other containers
             render_func = self._render_container
        # Add more dispatch rules here...
        # elif structural_role == "list_item": # Generic list item renderer?
             # render_func = self._render_list_item 
        # elif content_nature == "space_summary": # Maybe same as container?
             # render_func = self._render_container

        # --- Construct Output --- 
        output = "" 
        node_content_str = "" # Content generated by the specific renderer

        # Optional: Add opening tag if verbose style is enabled
        if use_verbose_tags:
             output += f"{indent_str}<{node_type} (Role: {structural_role or 'N/A'}, Nature: {content_nature or 'N/A'}) id='{node_id}'>\n"
             # Increase indent for content within tags
             content_indent_str = indent_str + "  "
        else:
             content_indent_str = indent_str # Use same indent for clean style

        # Call the chosen rendering function for the node's specific content
        try:
             node_content_str = render_func(node, attention, options, content_indent_str, node_info)
             # Prepend the content string to the main output
             output += node_content_str 
        except Exception as render_err:
             logger.error(f"Error calling renderer {render_func.__name__} for node {node_id}: {render_err}", exc_info=True)
             # Add error message, respecting indent
             output += f"{content_indent_str}>> Error rendering content for {node_info}: {render_err}\n"

        # Check attention (append after content for clarity)
        if node_id in attention:
             # Respect indent based on style
             output += f"{content_indent_str}  *ATTENTION: {attention[node_id].get('reason', '')}*\n"

        # Render children recursively (always use the main dispatcher)
        # Only render children if the current node isn't a specific content type 
        # that shouldn't have its VEIL children rendered (like a chat message itself)
        if children and render_func not in [self._render_chat_message]: # Add other terminal renderers here
            rendered_children_output = ""
            children_to_render = children # TODO: Apply filtering/limiting here
            # TODO: Sort children based on timestamp or other properties?
            # Example Sort (if timestamp exists): 
            # children_to_render.sort(key=lambda c: c.get('properties', {}).get('timestamp', 0))
            
            for child_node in children_to_render:
                 # Pass options down for consistent rendering style
                 rendered_children_output += self._render_veil_node_to_string(child_node, attention, options, indent + 1)
            
            # Only add if children produced output (avoid empty <Children> sections)
            if rendered_children_output: 
                 # Add children output respecting style
                 # In verbose mode, children are naturally indented inside the parent tag
                 # In clean mode, they follow the parent's rendered content
                 output += rendered_children_output 
                 
        # Optional: Add closing tag if verbose style is enabled
        if use_verbose_tags:
             output += f"{indent_str}</{node_type}>\n"
                 
        return output # Return accumulated string

    async def process_llm_response(self, llm_response_text: str) -> List[Dict[str, Any]]:
        """
        Parses the LLM's response text to extract structured actions and content.

        Args:
            llm_response_text: The raw text output from the LLM.

        Returns:
            A list of action dictionaries. Each dictionary should conform to a 
            structure that AgentLoopComponent can dispatch.
            Example action structure:
            {
                "action_name": "some_tool_or_verb",
                "target_element_id": "element_id_to_act_on", (optional, for element-specific actions)
                "target_space_id": "space_id_of_target_element", (optional, defaults to InnerSpace)
                "target_module": "module_name_for_external_action", (optional, for adapter actions)
                "parameters": { "param1": "value1", ... } 
            }
        """
        logger.debug(f"[{self.owner.id}] HUD processing LLM response: {llm_response_text[:100]}...")
        extracted_actions = []

        # Attempt to parse the response as JSON
        # This is a basic first pass. More sophisticated parsing (e.g., for VEIL-structured actions)
        # or natural language parsing could be added later.
        try:
            import json
            parsed_response = json.loads(llm_response_text)
            
            if isinstance(parsed_response, dict) and "actions" in parsed_response:
                actions_from_llm = parsed_response.get("actions")
                if isinstance(actions_from_llm, list):
                    for llm_action in actions_from_llm:
                        if isinstance(llm_action, dict):
                            # Basic transformation: assume llm_action structure matches our desired format.
                            # More mapping/validation might be needed here.
                            # e.g., mapping "action_type" from LLM to "action_name"
                            action_to_dispatch = {}
                            action_to_dispatch["action_name"] = llm_action.get("action_type") # or get("action_name")
                            action_to_dispatch["parameters"] = llm_action.get("parameters", {})
                            
                            # Add target_element_id, target_space_id, or target_module if present
                            if "target_element_id" in llm_action:
                                action_to_dispatch["target_element_id"] = llm_action["target_element_id"]
                            if "target_space_id" in llm_action:
                                action_to_dispatch["target_space_id"] = llm_action["target_space_id"]
                            if "target_module" in llm_action:
                                action_to_dispatch["target_module"] = llm_action["target_module"]

                            if action_to_dispatch.get("action_name"): # Must have an action name
                                extracted_actions.append(action_to_dispatch)
                                logger.info(f"Extracted action: {action_to_dispatch}")
                            else:
                                logger.warning(f"Skipping LLM action due to missing action_name: {llm_action}")
                        else:
                            logger.warning(f"Skipping non-dict item in LLM actions list: {llm_action}")
                else:
                    logger.warning("LLM response JSON had 'actions' key, but it was not a list.")
            else:
                # No 'actions' key found, or not a dict. Try other parsing? For now, assume no actions.
                logger.debug("LLM response not a dict or no 'actions' key found. No structured actions extracted.")

            # TODO: Handle other parts of the LLM response, e.g., "response_to_user" or free text
            # This content might need to be placed into the DAG as a new message/event.

        except json.JSONDecodeError:
            logger.warning(f"LLM response was not valid JSON: {llm_response_text[:200]}...")
            # Fallback: Treat the whole response as a potential natural language command?
            # For now, we don't parse actions from non-JSON.
        except Exception as e:
            logger.error(f"Error processing LLM response in HUD: {e}", exc_info=True)

        if not extracted_actions:
            logger.info(f"[{self.owner.id}] No actions extracted from LLM response.")
        
        return extracted_actions

    # Optional summarization helper
    # def _summarize_context(self, context: str) -> str: ...

    # Other potential methods:
    # - get_focused_context(element_id)
    # - get_context_summary()
