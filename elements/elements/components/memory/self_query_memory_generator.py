import logging
import time
import json
from typing import Dict, Any, Optional, List

# Base component and dependencies
from ..base_component import Component
from ..hud_component import HUDComponent
from ..context_manager_component import ContextManagerComponent
from .structured_memory_component import StructuredMemoryComponent

# Type hints
from ...llm.provider_interface import LLMProviderInterface, LLMMessage, LLMResponse
from ...base import BaseElement

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Placeholder Prompts --- 
# These should be loaded from config or constants
PRIMER_CONTENT_SQ = """(Placeholder: Primer content about digital minds, context, memory etc.)"""
SELF_QUERY_PROMPT = "Considering the preceding conversation snippet marked with <to_remember>, select the most relevant and important information, events, or insights from that period. Summarize these key points concisely for your future reference, focusing on what is essential to retain."

class SelfQueryMemoryGenerationComponent(Component):
    """
    Generates structured memories using a simple self-query approach.
    The agent model is prompted once to summarize the key points of a chunk.
    Relies on ContextManager for context/tail and StructuredMemoryStore for saving.
    """

    COMPONENT_TYPE: str = "memory_generator.self_query"
    DEPENDENCIES: List[str] = [
        ContextManagerComponent.COMPONENT_TYPE,
        StructuredMemoryComponent.COMPONENT_TYPE,
        HUDComponent.COMPONENT_TYPE
    ]

    def __init__(self, element: Optional[BaseElement] = None,
                 primer_content: str = PRIMER_CONTENT_SQ,
                 agent_model_override: Optional[str] = None, # Uses HUD default if None
                 tail_token_limit: int = 30000,
                 **kwargs):
        """
        Initializes the self-query memory generator.
        
        Args:
            element: The Element this component is attached to (typically InnerSpace).
            primer_content: The static primer text prepended to LLM contexts.
            agent_model_override: Specific model to use for the self-query step.
            tail_token_limit: Token limit for the history tail provided by ContextManager.
            **kwargs: Passthrough for BaseComponent.
        """
        super().__init__(element, **kwargs)
        self._state = {
            "primer_content": primer_content,
            "agent_model_override": agent_model_override,
            "tail_token_limit": tail_token_limit,
            "last_generation_stats": None
        }
        logger.info(f"SelfQueryMemoryGenerationComponent initialized for element {element.id if element else 'None'}")

    # --- Helper to get dependencies --- 
    def _get_deps(self) -> tuple[Optional[ContextManagerComponent], Optional[StructuredMemoryComponent], Optional[HUDComponent]]:
        if not self.element:
            logger.error("Cannot get dependencies: component not attached to an element.")
            return None, None, None
        ctx_mgr = self.element.get_component(ContextManagerComponent)
        mem_store = self.element.get_component(StructuredMemoryComponent)
        hud = self.element.get_component(HUDComponent)
        if not all([ctx_mgr, mem_store, hud]):
            logger.error("Missing required dependencies: ContextManager, StructuredMemoryStore, or HUDComponent.")
            return None, None, None
        if not hud._llm_provider:
             logger.error("HUDComponent is missing LLMProvider.")
             return None, None, None
        return ctx_mgr, mem_store, hud

    # --- Core Generation Logic --- 

    async def generate_memory_for_chunk(self, chunk_messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Executes the self-query memory generation process for a given chunk.

        Args:
            chunk_messages: A list of message dictionaries representing the chunk.
                            Assumes these are raw, uncompressed messages.
                            The list should contain at least one message.

        Returns:
            The memory_id of the generated and stored memory, or None on failure.
        """
        start_time = time.time()
        logger.info(f"Starting self-query memory generation for chunk of {len(chunk_messages)} messages.")
        if not chunk_messages:
            logger.warning("Cannot generate memory for empty chunk.")
            return None

        ctx_mgr, mem_store, hud = self._get_deps()
        if not all([ctx_mgr, mem_store, hud]):
            return None # Error already logged

        # --- Prepare Context Components --- 
        primer = self._state["primer_content"]
        chunk_start_timestamp = chunk_messages[0].get("timestamp")
        
        # 1. Get History Tail
        tail_messages = ctx_mgr.get_history_tail(
            before_timestamp=chunk_start_timestamp, 
            token_limit=self._state["tail_token_limit"]
        )
        logger.debug(f"Retrieved history tail with {len(tail_messages)} messages.")

        # 2. Format Chunk with Tags
        tagged_chunk_str = ""
        for msg in chunk_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tagged_chunk_str += f"<{role}_turn><to_remember>{json.dumps(content)}</to_remember></{role}_turn>\n"
        
        # 3. Prepare full context for the single LLM call
        base_context_str = f"{primer}\n\n--- History Tail ---\n{json.dumps(tail_messages)}\n\n--- Chunk to Remember ---\n{tagged_chunk_str}\n--- End Chunk --- \n\n"
        full_prompt = base_context_str + SELF_QUERY_PROMPT

        # --- LLM Call --- 
        llm_provider = hud._llm_provider
        agent_model = self._state.get("agent_model_override") or hud._state.get("model")

        try:
            logger.debug("Requesting Self-Query Summary...")
            summary_response = await llm_provider.complete(
                messages=[{"role": "user", "content": full_prompt}], model=agent_model
            )
            summary_content = summary_response.content if summary_response else "(Self-query summary failed)"
            logger.debug(f"Self-Query Summary: {summary_content[:100]}...")

            # --- Assemble Memory --- 
            memory_content = [
                # Simple structure: just the assistant's summary
                {"role": "assistant", "content": summary_content}
                # Could add a system marker if desired:
                # {"role": "system", "content": f"Summary of period starting {chunk_start_timestamp}:"},
                # {"role": "assistant", "content": summary_content}
            ]
            
            memory_metadata = {
                "generation_method": "self_query",
                "agent_model": agent_model,
            }
            source_info = {
                "type": "chunk_summary",
                "chunk_message_ids": [msg.get("uuid") or msg.get("id") for msg in chunk_messages],
                "chunk_start_timestamp": chunk_start_timestamp,
                "chunk_end_timestamp": chunk_messages[-1].get("timestamp")
            }

            final_memory_data = {
                "timestamp": int(time.time() * 1000), # Timestamp of memory creation
                "content": memory_content,
                "metadata": memory_metadata,
                "source_info": source_info
            }

            # Save the memory
            memory_id = mem_store.add_memory(final_memory_data)
            
            if memory_id:
                logger.info(f"Successfully generated and stored self-query memory {memory_id} in {(time.time() - start_time):.2f} seconds.")
                self._state["last_generation_stats"] = {"duration": time.time() - start_time, "memory_id": memory_id, "chunk_size": len(chunk_messages)}
                return memory_id
            else:
                 logger.error("Failed to store generated self-query memory.")
                 self._state["last_generation_stats"] = {"duration": time.time() - start_time, "error": "Storage failed", "chunk_size": len(chunk_messages)}
                 return None

        except Exception as e:
            logger.error(f"Error during self-query memory generation: {e}", exc_info=True)
            self._state["last_generation_stats"] = {"duration": time.time() - start_time, "error": str(e), "chunk_size": len(chunk_messages)}
            return None

    # TODO: Add methods to trigger memory generation? 