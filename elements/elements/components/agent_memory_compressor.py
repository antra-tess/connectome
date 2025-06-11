"""
AgentMemoryCompressor - Concrete Implementation

A concrete implementation of MemoryCompressor designed for agent memory management.
Features agent self-memorization via LLM reflection, file-based storage, and intelligent summarization.
"""

import asyncio
import logging
import json
import os
import aiofiles
from typing import Dict, List, Any, Optional
from datetime import datetime

from .memory_compressor_interface import MemoryCompressor, estimate_veil_tokens

logger = logging.getLogger(__name__)

class AgentMemoryCompressor(MemoryCompressor):
    """
    Agent-focused memory compressor with LLM-based self-memorization.
    
    Features:
    - Agent self-reflection via LLM calls
    - Agent decides what to remember from its perspective
    - File-based persistent storage with agent scoping
    - Intelligent, adaptive memory formation
    """
    
    def __init__(self, agent_id: str, token_limit: int = 4000, storage_base_path: str = "storage_data/memory_storage", llm_provider=None):
        # Agent-scoped storage path
        agent_storage_path = os.path.join(storage_base_path, "agents", agent_id)
        
        # Generate unique compressor ID using agent_id
        compressor_id = f"AgentMemoryCompressor_{agent_id}"
        
        super().__init__(
            token_limit=token_limit,
            storage_path=agent_storage_path, 
            compressor_id=compressor_id
        )
        
        self.agent_id = agent_id
        self.llm_provider = llm_provider
        
        # Ensure storage directory exists
        os.makedirs(agent_storage_path, exist_ok=True)
        
        logger.info(f"AgentMemoryCompressor initialized for agent {agent_id}")
    
    def set_llm_provider(self, llm_provider):
        """Set the LLM provider for agent reflection."""
        self.llm_provider = llm_provider
        logger.debug(f"LLM provider set for AgentMemoryCompressor {self.agent_id}")
    
    async def compress_nodes(self, 
                           raw_veil_nodes: List[Dict[str, Any]], 
                           element_ids: List[str],
                           compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compress VEIL nodes using agent self-memorization.
        
        The agent reflects on the experience and creates its own memory summary.
        """
        try:
            if not self.llm_provider:
                logger.warning(f"No LLM provider available for agent {self.agent_id}, falling back to basic analysis")
                return await self._fallback_compression(raw_veil_nodes, element_ids, compression_context)
            
            logger.info(f"AgentMemoryCompressor compressing {len(raw_veil_nodes)} nodes for elements {element_ids}")
            
            # Prepare content for agent reflection
            veil_content_for_reflection = await self._prepare_content_for_reflection(raw_veil_nodes)
            
            # NEW: Add raw VEIL nodes to compression context for multimodal extraction
            enhanced_compression_context = (compression_context or {}).copy()
            enhanced_compression_context["raw_veil_nodes"] = raw_veil_nodes
            
            # Get agent's self-reflection on the experience
            agent_memory_summary = await self._agent_reflect_on_experience(
                veil_content_for_reflection, 
                element_ids, 
                enhanced_compression_context
            )
            
            # Count tokens in original content
            total_tokens = estimate_veil_tokens(raw_veil_nodes)
            
            # Create memory metadata
            memory_metadata = {
                "agent_id": self.agent_id,
                "compression_approach": "agent_self_reflection",
                "original_node_count": len(raw_veil_nodes),
                "original_element_ids": element_ids,
                "token_count": total_tokens,
                "compression_timestamp": datetime.now().isoformat(),
                "compressor_type": self.__class__.__name__,
                "compression_context": compression_context or {},
                "content_fingerprint": compression_context.get("content_fingerprint") if compression_context else None
            }
            
            # Generate memory ID
            memory_id = self._generate_memory_id(element_ids)
            
            # Create memorized VEIL node
            memorized_node = {
                "veil_id": f"memorized_{memory_id}",
                "node_type": "memorized_content",
                "properties": {
                    "structural_role": "compressed_content",
                    "content_nature": "agent_memory",
                    "memory_id": memory_id,
                    "memory_summary": agent_memory_summary,
                    "original_element_ids": element_ids,
                    "original_node_count": len(raw_veil_nodes),
                    "token_count": total_tokens,
                    "compression_timestamp": datetime.now().isoformat(),
                    "compressor_type": self.__class__.__name__,
                    "compression_metadata": memory_metadata
                },
                "children": []
            }
            
            # Store memory persistently
            await self._store_memory_to_file(memory_id, {
                "memory_summary": agent_memory_summary,
                "metadata": memory_metadata,
                "memorized_node": memorized_node
            })
            
            # Update correlations
            self.add_correlation(element_ids, memory_id)
            
            logger.info(f"Created agent memory {memory_id}: {agent_memory_summary[:50]}... ({total_tokens} tokens)")
            
            return memorized_node
            
        except Exception as e:
            logger.error(f"Error in agent memory compression: {e}", exc_info=True)
            # Fallback to basic compression
            return await self._fallback_compression(raw_veil_nodes, element_ids, compression_context)
    
    async def _prepare_content_for_reflection(self, raw_veil_nodes: List[Dict[str, Any]]) -> str:
        """
        Prepare VEIL content in a format suitable for agent reflection.
        
        ENHANCED: Now includes attachment content analysis for multimodal memories.
        This renders the VEIL content in a clean, readable format for the agent to reflect on.
        """
        try:
            content_parts = []
            
            for node in raw_veil_nodes:
                props = node.get("properties", {})
                content_nature = props.get("content_nature", "")
                
                if content_nature == "chat_message":
                    # Format chat messages nicely
                    sender = props.get("sender_name", "Unknown")
                    text = props.get("text_content", "")
                    timestamp = props.get("timestamp_iso", props.get("timestamp", ""))
                    
                    message_line = f"[{timestamp}] {sender}: {text}"
                    
                    # NEW: Enhanced attachment info with content analysis
                    attachments = props.get("attachment_metadata", [])
                    if attachments:
                        attachment_descriptions = []
                        for att in attachments:
                            filename = att.get("filename", "attachment")
                            content_type = att.get("attachment_type", att.get("content_type", "unknown"))
                            
                            # Basic attachment description
                            att_desc = f"{filename} ({content_type})"
                            attachment_descriptions.append(att_desc)
                        
                        message_line += f" (attachments: {', '.join(attachment_descriptions)})"
                    
                    # NEW: Look for attachment content in child nodes
                    attachment_content_descriptions = await self._process_attachment_children_for_reflection(node)
                    if attachment_content_descriptions:
                        message_line += f"\n    Attachment content: {attachment_content_descriptions}"
                    
                    content_parts.append(message_line)
                    
                elif content_nature == "uplink_summary":
                    # Include uplink information
                    remote_name = props.get("remote_space_name", "Remote Space")
                    content_parts.append(f"[Uplink] Connected to {remote_name}")
                    
                elif props.get("structural_role") == "container":
                    # Include container context
                    element_name = props.get("element_name", "Element")
                    available_tools = props.get("available_tools", [])
                    
                    container_line = f"[Context] {element_name}"
                    if available_tools:
                        container_line += f" (tools available: {', '.join(available_tools)})"
                    content_parts.append(container_line)
                
                # Handle other content types as needed
                elif props.get("text_content"):
                    # Generic text content
                    content_parts.append(f"[Content] {props['text_content']}")
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error preparing content for reflection: {e}", exc_info=True)
            # Fallback to basic representation
            return str(raw_veil_nodes)
    
    async def _process_attachment_children_for_reflection(self, message_node: Dict[str, Any]) -> str:
        """
        NEW: Process attachment child nodes to extract content descriptions for agent reflection.
        
        Args:
            message_node: VEIL message node that may have attachment children
            
        Returns:
            String description of attachment content for agent reflection
        """
        try:
            children = message_node.get("children", [])
            attachment_descriptions = []
            
            for child in children:
                child_props = child.get("properties", {})
                if child.get("node_type") == "attachment_content_item":
                    filename = child_props.get("filename", "unknown_file")
                    content_nature = child_props.get("content_nature", "unknown")
                    attachment_id = child_props.get("attachment_id", "unknown")
                    
                    # Create description based on content type
                    if "image" in content_nature.lower():
                        # For images, we could analyze content if available
                        attachment_descriptions.append(f"{filename} (image content available for analysis)")
                    elif "text" in content_nature.lower() or content_nature in ["application/pdf", "text/plain"]:
                        # For text/document content
                        attachment_descriptions.append(f"{filename} (document content available for analysis)")
                    else:
                        # For other file types
                        attachment_descriptions.append(f"{filename} ({content_nature} content)")
            
            return "; ".join(attachment_descriptions) if attachment_descriptions else ""
            
        except Exception as e:
            logger.error(f"Error processing attachment children for reflection: {e}", exc_info=True)
            return ""
    
    async def _agent_reflect_on_experience(self, 
                                         veil_content: str, 
                                         element_ids: List[str],
                                         compression_context: Optional[Dict[str, Any]]) -> str:
        """
        Have the agent reflect on its experience and create its own memory summary.
        
        ENHANCED: Now handles multimodal content - agent is aware of images and documents.
        This is the core of agent self-memorization - the agent decides what to remember.
        """
        try:
            # NEW: Check if this content includes multimodal attachments
            has_attachments = "Attachment content:" in veil_content or "attachments:" in veil_content
            
            # Prepare context information
            element_context = ", ".join(element_ids) if len(element_ids) > 1 else element_ids[0]
            
            # Focus context if available
            focus_info = ""
            if compression_context and compression_context.get("focus_element_id"):
                focus_info = f"\nNote: You were focusing on {compression_context['focus_element_id']} during this interaction."
            
            # Compression reason
            compression_reason = compression_context.get("compression_reason", "memory management") if compression_context else "memory management"
            
            # NEW: Enhanced reflection prompt for multimodal content
            if has_attachments:
                reflection_prompt = f"""You are an AI agent reflecting on your recent experience that included visual and document content. Look at this interaction and decide what you should remember about it for future reference.

EXPERIENCE TO REFLECT ON:
{veil_content}

CONTEXT:
- This experience occurred in: {element_context}
- Compression reason: {compression_reason}{focus_info}

MULTIMODAL REFLECTION TASK:
Create a concise memory summary that captures what was important about this experience from YOUR perspective as an AI agent. Since this interaction included attachments (images, documents, files), pay special attention to:

CONTENT ANALYSIS:
- What happened that you should remember
- Who was involved and their roles  
- What images/documents were shared and their relevance
- How the visual/document content related to the conversation
- Any important outcomes or decisions
- Context that would be useful for future interactions

MULTIMODAL CONTEXT:
- Describe what the attachments represented (diagrams, documents, images, etc.)
- Explain how the attachments connected to the conversation topic
- Note any important visual or document information discussed

Keep the summary brief but informative - this will be your memory of this multimodal experience.

MEMORY SUMMARY:"""
            else:
                # Standard text-only reflection prompt
                reflection_prompt = f"""You are an AI agent reflecting on your recent experience. Look at this interaction and decide what you should remember about it for future reference.

EXPERIENCE TO REFLECT ON:
{veil_content}

CONTEXT:
- This experience occurred in: {element_context}
- Compression reason: {compression_reason}{focus_info}

REFLECTION TASK:
Create a concise memory summary that captures what was important about this experience from YOUR perspective as an AI agent. Focus on:
- What happened that you should remember
- Who was involved and their roles
- Any important outcomes or decisions
- Context that would be useful for future interactions

Keep the summary brief but informative - this will be your memory of this experience.

MEMORY SUMMARY:"""

            # NEW: Create multimodal LLM message if attachments detected
            from llm.provider_interface import LLMMessage
            
            if has_attachments:
                # Extract actual multimodal content for the agent
                multimodal_content = await self._extract_multimodal_content_for_agent(compression_context.get("raw_veil_nodes", []))
                
                if multimodal_content and multimodal_content.get("attachments"):
                    # Create multimodal message structure
                    reflection_message = LLMMessage("user", {
                        "text": reflection_prompt,
                        "attachments": multimodal_content["attachments"]
                    })
                    logger.info(f"Agent {self.agent_id} reflecting on multimodal content: {len(multimodal_content['attachments'])} attachments")
                else:
                    # Fallback to text-only if attachment extraction failed
                    reflection_message = LLMMessage("user", reflection_prompt)
                    logger.warning(f"Agent {self.agent_id} multimodal extraction failed, using text-only reflection")
            else:
                # Text-only reflection
                reflection_message = LLMMessage("user", reflection_prompt)
            
            # Get agent's reflection via LLM
            llm_response = self.llm_provider.complete(messages=[reflection_message], tools=None)
            
            if llm_response and llm_response.content:
                agent_summary = llm_response.content.strip()
                
                # NEW: Log multimodal memory formation
                if has_attachments:
                    logger.info(f"Agent {self.agent_id} created multimodal memory: {agent_summary[:100]}...")
                else:
                    logger.debug(f"Agent {self.agent_id} reflected: {agent_summary[:100]}...")
                
                return agent_summary
            else:
                logger.warning(f"LLM reflection failed for agent {self.agent_id}, using fallback")
                return await self._create_fallback_summary(veil_content, element_ids, has_attachments)
                
        except Exception as e:
            logger.error(f"Error during agent reflection: {e}", exc_info=True)
            return await self._create_fallback_summary(veil_content, element_ids, False)
    
    async def _create_fallback_summary(self, veil_content: str, element_ids: List[str], has_attachments: bool = False) -> str:
        """
        Create a basic fallback summary when LLM reflection fails.
        
        ENHANCED: Now aware of multimodal content.
        """
        lines = veil_content.split('\n')
        message_lines = [line for line in lines if '] ' in line and ': ' in line]
        attachment_lines = [line for line in lines if 'Attachment content:' in line or 'attachments:' in line]
        
        if message_lines:
            base_summary = f"Conversation with {len(message_lines)} messages in {', '.join(element_ids)}"
            
            # NEW: Add attachment context to fallback
            if has_attachments or attachment_lines:
                base_summary += " (with attachments)"
                
            return base_summary
        else:
            return f"Content from {', '.join(element_ids)}"
    
    async def _fallback_compression(self, 
                                  raw_veil_nodes: List[Dict[str, Any]], 
                                  element_ids: List[str],
                                  compression_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback compression when LLM is not available."""
        logger.info(f"Using fallback compression for {len(raw_veil_nodes)} nodes")
        
        # Simple analysis
        message_count = 0
        participants = set()
        
        for node in raw_veil_nodes:
            props = node.get("properties", {})
            if props.get("content_nature") == "chat_message":
                message_count += 1
                sender = props.get("sender_name")
                if sender:
                    participants.add(sender)
        
        # Create basic summary
        if message_count > 0:
            if len(participants) <= 2:
                summary = f"Conversation with {', '.join(participants)}: {message_count} messages"
            else:
                summary = f"Group conversation: {message_count} messages"
        else:
            summary = f"Content from {', '.join(element_ids)}"
        
        # Count tokens
        total_tokens = estimate_veil_tokens(raw_veil_nodes)
        memory_id = self._generate_memory_id(element_ids)
        
        # Create memorized node
        memorized_node = {
            "veil_id": f"memorized_{memory_id}",
            "node_type": "memorized_content",
            "properties": {
                "structural_role": "compressed_content",
                "content_nature": "agent_memory",
                "memory_id": memory_id,
                "memory_summary": summary,
                "original_element_ids": element_ids,
                "original_node_count": len(raw_veil_nodes),
                "token_count": total_tokens,
                "compression_timestamp": datetime.now().isoformat(),
                "compressor_type": f"{self.__class__.__name__}_fallback"
            },
            "children": []
        }
        
        # Store and track
        await self._store_memory_to_file(memory_id, {
            "memory_summary": summary,
            "metadata": {"fallback": True, "agent_id": self.agent_id},
            "memorized_node": memorized_node
        })
        
        self.add_correlation(element_ids, memory_id)
        
        logger.info(f"Created fallback memory {memory_id}: {summary}")
        return memorized_node
    
    # Storage methods remain the same
    async def _store_memory_to_file(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Store memory data to agent-scoped file."""
        try:
            file_path = os.path.join(self.storage_path, f"{memory_id}.json")
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(memory_data, indent=2))
            
            logger.debug(f"Stored memory {memory_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory to file: {e}", exc_info=True)
            return False
    
    async def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load memory data from agent-scoped file."""
        try:
            file_path = os.path.join(self.storage_path, f"{memory_id}.json")
            
            if not os.path.exists(file_path):
                return None
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Error loading memory from file: {e}", exc_info=True)
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from agent-scoped file storage."""
        try:
            file_path = os.path.join(self.storage_path, f"{memory_id}.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted memory file {file_path}")
            
            # Remove from correlations - find element_ids for this memory_id
            for element_set, corr_memory_id in list(self._correlations.items()):
                if corr_memory_id == memory_id:
                    self.remove_correlation(list(element_set))
                    break
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory file: {e}", exc_info=True)
            return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent memory storage."""
        try:
            memory_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
            total_size = sum(os.path.getsize(os.path.join(self.storage_path, f)) for f in memory_files)
            
            # Get base statistics and add agent-specific info
            base_stats = super().get_memory_stats()
            
            return {
                "agent_id": self.agent_id,
                "total_memories": len(memory_files),
                "storage_size_mb": total_size / (1024 * 1024),
                "storage_path": self.storage_path,
                "correlations_tracked": len(self._correlations),
                **base_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}", exc_info=True)
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up agent memory compressor resources."""
        logger.info(f"Cleaning up AgentMemoryCompressor for agent {self.agent_id}")
        super().cleanup()
    
    async def compress(self, 
                      raw_veil_nodes: List[Dict[str, Any]], 
                      element_ids: List[str],
                      compression_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compress VEIL nodes using agent self-memorization.
        
        The agent reflects on the experience and creates its own memory summary.
        """
        return await self.compress_nodes(raw_veil_nodes, element_ids, compression_context)
    
    async def estimate_tokens(self, raw_veil_nodes: List[Dict[str, Any]]) -> int:
        """Estimate tokens using the interface utility."""
        return estimate_veil_tokens(raw_veil_nodes)
    
    async def store_memory(self, memory_id: str, memorized_veil_node: Dict[str, Any]) -> bool:
        """Store memory to agent-scoped file (interface compliance)."""
        try:
            memory_data = {
                "memory_summary": memorized_veil_node["properties"]["memory_summary"],
                "metadata": memorized_veil_node["properties"].get("compression_metadata", {}),
                "memorized_node": memorized_veil_node
            }
            return await self._store_memory_to_file(memory_id, memory_data)
        except Exception as e:
            logger.error(f"Error in store_memory interface: {e}", exc_info=True)
            return False

    async def _extract_multimodal_content_for_agent(self, raw_veil_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Extract actual multimodal content for the agent to analyze during reflection.
        
        OPTIMIZED: Efficiently searches through the provided VEIL nodes for attachment content,
        using both direct node checking and recursive child searching.
        
        Args:
            raw_veil_nodes: List of VEIL nodes to extract content from
             
        Returns:
            Dictionary with 'attachments' key containing LiteLLM-compatible attachment data
        """
        try:
            attachments = []
            
            # OPTIMIZED: First check if any nodes are directly attachment nodes
            direct_attachment_nodes = [
                node for node in raw_veil_nodes
                if isinstance(node, dict) and node.get("node_type") == "attachment_content_item"
            ]
            
            if direct_attachment_nodes:
                logger.debug(f"Found {len(direct_attachment_nodes)} direct attachment nodes")
                
                for attachment_node in direct_attachment_nodes:
                    # Direct attachment processing
                    attachment_content = await self._process_attachment_for_llm(attachment_node)
                    if attachment_content:
                        attachments.append(attachment_content)
                        props = attachment_node.get("properties", {})
                        filename = props.get("filename", "unknown")
                        logger.debug(f"Extracted direct attachment for agent: {filename}")
            
            # OPTIMIZED: Then recursively search for attachment children in all nodes
            for node in raw_veil_nodes:
                child_attachments = self._search_node_children_for_attachments(node)
                for attachment_node in child_attachments:
                    # Avoid duplicates (in case a node was both direct and child)
                    attachment_id = attachment_node.get("properties", {}).get("attachment_id")
                    
                    # Check if we already processed this attachment
                    already_processed = any(
                        att.get("attachment_id") == attachment_id 
                        for att in attachments 
                        if isinstance(att, dict) and att.get("attachment_id")
                    )
                    
                    if not already_processed:
                        attachment_content = await self._process_attachment_for_llm(attachment_node)
                        if attachment_content:
                            # Add attachment_id for deduplication tracking
                            if attachment_id:
                                attachment_content["attachment_id"] = attachment_id
                            attachments.append(attachment_content)
                            props = attachment_node.get("properties", {})
                            filename = props.get("filename", "unknown")
                            logger.debug(f"Extracted child attachment for agent: {filename}")
            
            logger.info(f"Extracted {len(attachments)} multimodal attachments for agent reflection via optimized search")
            
            return {
                "attachments": attachments
            }
            
        except Exception as e:
            logger.error(f"Error extracting multimodal content for agent: {e}", exc_info=True)
            return {}
    
    def _search_node_children_for_attachments(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        NEW: Efficiently search through a node's children for attachment content.
        
        Args:
            node: VEIL node to search through
            
        Returns:
            List of attachment nodes found in children
        """
        try:
            found_attachments = []
            
            if not isinstance(node, dict):
                return found_attachments
            
            # Recursively search children
            children = node.get("children", [])
            for child in children:
                if not isinstance(child, dict):
                    continue
                
                # Check if this child is an attachment content item
                if child.get("node_type") == "attachment_content_item":
                    logger.debug(f"Found attachment child node for agent reflection: {child.get('veil_id')}")
                    found_attachments.append(child)
                
                # Recursively search this child's children
                found_attachments.extend(self._search_node_children_for_attachments(child))
            
            return found_attachments
            
        except Exception as e:
            logger.error(f"Error searching node children for attachments: {e}", exc_info=True)
            return []
    
    async def _process_attachment_for_llm(self, attachment_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        NEW: Process an attachment VEIL node into LiteLLM-compatible format for agent analysis.
        
        This converts VEIL attachment nodes into the format needed for multimodal LLM calls
        so the agent can actually see images and analyze document content.
        
        Args:
            attachment_node: VEIL node representing attachment content
            
        Returns:
            LiteLLM-compatible content part or None if not processable
        """
        try:
            props = attachment_node.get("properties", {})
            content_nature = props.get("content_nature", "unknown")
            filename = props.get("filename", "unknown_file")
            attachment_id = props.get("attachment_id")
            
            if "image" in content_nature.lower():
                # For images, try to get base64 content
                logger.debug(f"Processing image attachment for agent: {filename}")
                
                content = await self._get_attachment_content_for_agent(attachment_id, attachment_node)
                if content and isinstance(content, str):
                    # Format for LiteLLM multimodal
                    return {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{content_nature.split('/')[-1] if '/' in content_nature else 'png'};base64,{content}"
                        }
                    }
                else:
                    logger.warning(f"Could not extract image content for {attachment_id}")
                    return None
                    
            elif "text" in content_nature.lower() or content_nature in ["application/pdf", "text/plain"]:
                # For text/document files, include content directly  
                logger.debug(f"Processing text/document attachment for agent: {filename}")
                
                content = await self._get_attachment_content_for_agent(attachment_id, attachment_node)
                if content:
                    return {
                        "type": "text",
                        "text": f"[Document: {filename}]\n{content}"
                    }
                else:
                    logger.warning(f"Could not extract document content for {attachment_id}")
                    return None
            else:
                # For other file types, provide descriptive text
                logger.debug(f"Unsupported attachment type for agent analysis: {content_nature}")
                return {
                    "type": "text", 
                    "text": f"[Attachment: {filename} (Type: {content_nature}) - Content not directly analyzable by agent]"
                }
                
        except Exception as e:
            logger.error(f"Error processing attachment for LLM: {e}", exc_info=True)
            return None
    
    async def _get_attachment_content_for_agent(self, attachment_id: str, attachment_node: Dict[str, Any]) -> Optional[str]:
        """
        NEW: Get the actual attachment content for agent analysis.
        
        SIMPLIFIED: With VEIL's clean architecture, content is directly available in node properties.
        
        Args:
            attachment_id: ID of the attachment to retrieve
            attachment_node: VEIL attachment_content_item node with attachment data
            
        Returns:
            Attachment content as string (base64 for images, text for documents), or None
        """
        try:
            props = attachment_node.get("properties", {})
            filename = props.get("filename", "unknown_file")
            content_nature = props.get("content_nature", "unknown")
            
            # NEW: Get content directly from VEIL node properties (matches HUD approach)
            content = props.get("content")
            content_available = props.get("content_available", False)
            
            if content:
                logger.debug(f"Found direct content in VEIL node for agent analysis: {filename}")
                return content
            
            # Log when content should be available but isn't found
            if content_available:
                logger.warning(f"Content marked as available for {filename} but not found in VEIL node properties")
            else:
                logger.debug(f"Content not marked as available for {filename}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving attachment content for agent: {e}", exc_info=True)
            return None 