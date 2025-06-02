"""
Element ID Generator Utility

Provides consistent, deterministic element ID generation for chat elements
across the entire system. Used by both ChatManagerComponent and ExternalEventRouter
to ensure target element IDs match actual element IDs.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ElementIdGenerator:
    """
    Utility class for generating deterministic element IDs based on conversation parameters.
    
    This ensures that:
    1. Chat elements always get the same ID for the same conversation
    2. Event routing targets match actual element IDs  
    3. DM and channel elements follow unified naming conventions
    4. System restarts don't create duplicate elements
    """
    
    @staticmethod
    def generate_chat_element_id(
        adapter_id: str, 
        conversation_id: str, 
        is_dm: bool, 
        owner_space_id: Optional[str] = None
    ) -> str:
        """
        Generate a deterministic chat element ID based on conversation parameters.
        
        Args:
            adapter_id: The adapter ID (e.g., 'zulip_adapter', 'discord')
            conversation_id: The conversation/channel ID from the external platform
            is_dm: Whether this is a direct message (True) or channel message (False)
            owner_space_id: ID of the owning space (for additional uniqueness)
            
        Returns:
            Deterministic element ID (e.g., 'dm_elem_zulip_adapter_737_756_b0dab7c5')
        """
        # Determine session type
        session_type = "dm" if is_dm else "chat"
        
        # Sanitize conversation_id for use in element ID (remove special characters)
        safe_conv_id = ElementIdGenerator._sanitize_conversation_id(conversation_id)
        
        # Create deterministic hash from the key parameters
        hash_input = f"{session_type}_{adapter_id}_{safe_conv_id}"
        if owner_space_id:
            hash_input += f"_{owner_space_id}"
            
        hash_object = hashlib.md5(hash_input.encode())
        short_hash = hash_object.hexdigest()[:8]  # Use first 8 characters
        
        # Construct the element ID
        element_id = f"{session_type}_elem_{adapter_id}_{safe_conv_id}_{short_hash}"
        
        # Ensure element ID is not too long (some systems have limits)
        if len(element_id) > 100:
            # If too long, use more hash and less original text
            hash_object = hashlib.md5(hash_input.encode())
            full_hash = hash_object.hexdigest()[:16]
            element_id = f"{session_type}_elem_{full_hash}"
            
        logger.debug(f"Generated chat element ID: {hash_input} -> {element_id}")
        return element_id
    
    @staticmethod
    def generate_target_element_id(
        adapter_id: str, 
        conversation_id: str, 
        is_dm: bool, 
        owner_space_id: Optional[str] = None
    ) -> str:
        """
        Generate the target element ID for event routing.
        
        This is an alias for generate_chat_element_id() to make it clear
        when we're generating IDs for event targeting vs element creation.
        
        Args:
            adapter_id: The adapter ID
            conversation_id: The conversation/channel ID  
            is_dm: Whether this is a direct message
            owner_space_id: ID of the owning space
            
        Returns:
            Target element ID for event routing
        """
        return ElementIdGenerator.generate_chat_element_id(
            adapter_id, conversation_id, is_dm, owner_space_id
        )
    
    @staticmethod
    def _sanitize_conversation_id(conversation_id: str) -> str:
        """
        Sanitize conversation ID for use in element IDs.
        
        Removes or replaces characters that might cause issues in element IDs.
        
        Args:
            conversation_id: Raw conversation ID from external platform
            
        Returns:
            Sanitized conversation ID safe for use in element IDs
        """
        if not conversation_id:
            return "unknown"
            
        # Replace problematic characters with underscores
        sanitized = conversation_id.replace(':', '_') \
                                 .replace('@', '_') \
                                 .replace('.', '_') \
                                 .replace('/', '_') \
                                 .replace(' ', '_') \
                                 .replace('#', '_') \
                                 .replace('?', '_') \
                                 .replace('&', '_') \
                                 .replace('=', '_') \
                                 .replace('+', '_') \
                                 .replace('%', '_')
        
        # Remove any remaining non-alphanumeric characters except underscores and hyphens
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in ['_', '-'])
        
        # Ensure it doesn't start with a number (some systems don't like that)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"conv_{sanitized}"
            
        # Limit length to prevent overly long IDs
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
            
        return sanitized or "unknown"
    
    @staticmethod
    def parse_element_id(element_id: str) -> Optional[dict]:
        """
        Parse an element ID back into its component parts (for debugging/analysis).
        
        Args:
            element_id: The element ID to parse
            
        Returns:
            Dictionary with parsed components or None if parsing fails
        """
        try:
            # Expected format: {session_type}_elem_{adapter_id}_{sanitized_conv_id}_{hash}
            # Actual examples:
            # - "dm_elem_zulip_adapter_conv_737_756_f630cf12" -> ['dm', 'elem', 'zulip', 'adapter', 'conv', '737', '756', 'f630cf12']
            # - "chat_elem_zulip_adapter_general_804b6f2e" -> ['chat', 'elem', 'zulip', 'adapter', 'general', '804b6f2e']
            
            parts = element_id.split('_')
            if len(parts) < 4 or parts[1] != "elem":
                return None
                
            session_type = parts[0]  # 'dm' or 'chat'
            hash_part = parts[-1]    # Last part is the hash (typically 8 hex chars)
            
            # Middle parts contain adapter and conversation
            middle_parts = parts[2:-1]  # Exclude session_type, 'elem', and hash
            
            if not middle_parts:
                return None
            
            # Strategy: Look for known adapter patterns and conversation patterns
            # Most adapters end with '_adapter', conversations often start with 'conv_' or are simple names
            
            adapter_parts = []
            conv_parts = []
            
            # Find the split point between adapter and conversation
            found_split = False
            for i in range(len(middle_parts)):
                # Check if this could be the end of an adapter name
                if i + 1 < len(middle_parts):
                    current_part = middle_parts[i]
                    next_part = middle_parts[i + 1] if i + 1 < len(middle_parts) else None
                    
                    # Case 1: Current part is 'adapter' (common adapter pattern)
                    if current_part == 'adapter':
                        adapter_parts = middle_parts[:i + 1]
                        conv_parts = middle_parts[i + 1:]
                        found_split = True
                        break
                    
                    # Case 2: Next part is 'conv' (conversation starts)
                    if next_part == 'conv':
                        adapter_parts = middle_parts[:i + 1]
                        conv_parts = middle_parts[i + 1:]
                        found_split = True
                        break
            
            # Fallback: if no clear split found, assume first 2 parts are adapter
            if not found_split:
                if len(middle_parts) >= 2:
                    adapter_parts = middle_parts[:2]  # e.g., ['zulip', 'adapter']
                    conv_parts = middle_parts[2:]     # Rest is conversation
                else:
                    adapter_parts = middle_parts
                    conv_parts = []
            
            adapter_id = '_'.join(adapter_parts) if adapter_parts else "unknown"
            
            # Clean up conversation ID
            if conv_parts:
                conv_id = '_'.join(conv_parts)
                # Remove 'conv_' prefix if present
                if conv_id.startswith('conv_'):
                    conv_id = conv_id[5:]
                elif conv_id == 'conv':
                    conv_id = "unknown"
            else:
                conv_id = "unknown"
                
            return {
                "session_type": session_type,
                "is_dm": session_type == "dm",
                "adapter_id": adapter_id,
                "conversation_id": conv_id,
                "hash": hash_part,
                "original_id": element_id
            }
        except Exception as e:
            logger.warning(f"Failed to parse element ID '{element_id}': {e}")
            
        return None


# Convenience functions for backward compatibility and ease of use
def generate_chat_element_id(adapter_id: str, conversation_id: str, is_dm: bool, owner_space_id: Optional[str] = None) -> str:
    """Convenience function for ElementIdGenerator.generate_chat_element_id()"""
    return ElementIdGenerator.generate_chat_element_id(adapter_id, conversation_id, is_dm, owner_space_id)


def generate_target_element_id(adapter_id: str, conversation_id: str, is_dm: bool, owner_space_id: Optional[str] = None) -> str:
    """Convenience function for ElementIdGenerator.generate_target_element_id()"""
    return ElementIdGenerator.generate_target_element_id(adapter_id, conversation_id, is_dm, owner_space_id) 