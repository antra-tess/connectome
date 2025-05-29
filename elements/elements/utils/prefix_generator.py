"""
Shared prefix generation utility for consistent tool naming across components.

This module provides a centralized way to generate short, consistent prefixes
for element IDs to avoid tool name conflicts and stay within Anthropic's 64-char limit.
"""

import hashlib
import re
from typing import Dict, Any

def create_short_element_prefix(element_id: str) -> str:
    """
    Create a short prefix for tool names from element ID to fit Anthropic's 64-char limit.
    
    This is the canonical implementation that should be used by both HUD and AgentLoop
    to ensure consistency.
    
    For DM elements like 'dm_elem_discord_adapter_1_alice_smith_a1b2c3d4',
    extracts meaningful parts to create something like 'alice_a1b2' or 'dm_a1b2c3d4'.
    For uplink elements with very long IDs, creates concise hashed prefixes.
    
    Args:
        element_id: Full element ID
        
    Returns:
        Short prefix (max 12 characters to leave plenty of room for tool names) that matches ^[a-zA-Z0-9_-]+$
    """
    def sanitize_for_anthropic(text: str) -> str:
        """Remove or replace characters not allowed by Anthropic's regex ^[a-zA-Z0-9_-]+$"""
        # Replace invalid characters with underscores, then remove duplicate underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
        # Remove duplicate underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized
    
    # Create a hash for uniqueness
    hash_obj = hashlib.md5(element_id.encode())
    short_hash = hash_obj.hexdigest()[:6]  # 6 chars for uniqueness
    
    # For DM elements, try to extract user identifier
    if element_id.startswith('dm_elem_'):
        parts = element_id.split('_')
        if len(parts) >= 4:
            # parts: ['dm', 'elem', 'adapter_id', 'user_part', 'uuid_part', ...]
            user_part = parts[-2] if len(parts) > 3 else 'dm'  # Second to last part
            user_short = sanitize_for_anthropic(user_part)[:4] if user_part else 'dm'
            return f"{user_short}_{short_hash[:4]}"[:12]  # Max 12 chars
    
    # For uplink elements (which can be very long), use a concise approach
    if 'uplink' in element_id.lower():
        # Extract meaningful parts: uplink, any agent identifier
        if 'shared' in element_id:
            return f"up_sh_{short_hash[:4]}"[:12]  # "up_sh_" + 4 char hash
        elif 'agent' in element_id:
            return f"up_ag_{short_hash[:4]}"[:12]  # "up_ag_" + 4 char hash
        else:
            return f"up_{short_hash[:6]}"[:12]    # "up_" + 6 char hash
    
    # For shared space elements
    if element_id.startswith('shared_'):
        parts = element_id.split('_')
        if len(parts) >= 2:
            # Try to extract adapter info
            adapter_hint = parts[1][:2] if len(parts) > 1 else 'sh'
            return f"{adapter_hint}_{short_hash[:6]}"[:12]
        else:
            return f"sh_{short_hash[:6]}"[:12]
    
    # For other elements, use a more aggressive shortening approach
    if '_' in element_id:
        # Extract first meaningful part but keep it very short
        meaningful_part = element_id.split('_')[0][:3]  # Only 3 chars max
        meaningful_part = sanitize_for_anthropic(meaningful_part)
        return f"{meaningful_part}_{short_hash[:6]}"[:12]
    else:
        # Sanitize the entire element_id but keep it very short
        sanitized_id = sanitize_for_anthropic(element_id)[:3]
        return f"{sanitized_id}_{short_hash[:6]}"[:12] if sanitized_id else f"el_{short_hash[:8]}"[:12] 