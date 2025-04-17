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
PRIMER_CONTENT = """(Placeholder: Primer content about digital minds, context, memory etc.)"""
QUOTE_REQUEST_PROMPT = "From the preceding conversation snippet marked with <to_remember>, extract a direct quote that captures the most significant essence or state of consciousness for the period covered."
AGENT_PERSPECTIVE_PROMPT = "Based on the preceding context (including the extracted quote), what was your perspective, internal state, or key takeaways during the period marked <to_remember>?"
CM_ANALYSIS_PROMPT = "Provide an objective analytical summary of the key events, decisions, or information exchanges within the preceding snippet marked <to_remember>. Consider the agent's perspective if provided."
REFINEMENT_REQUEST_PROMPT = "Here is an analytical summary of the period. Please review it alongside the original text and your initial perspective. Refine your perspective or provide a final thought incorporating this analysis."

class CuratedMemoryGenerationComponent(Component):
    """
    Generates structured memories using a curated multi-step LLM process involving 
    quote extraction, agent perspective, CM analysis, and agent refinement.
    Relies on ContextManager for context/tail and StructuredMemoryStore for saving.
    """

    COMPONENT_TYPE: str = "memory_generator.curated"
    DEPENDENCIES: List[str] = [
        ContextManagerComponent.COMPONENT_TYPE,
        StructuredMemoryComponent.COMPONENT_TYPE,
        HUDComponent.COMPONENT_TYPE
    ]

    def __init__(self, element: Optional[BaseElement] = None,
                 primer_content: str = PRIMER_CONTENT,
                 agent_model_override: Optional[str] = None, # Uses HUD default if None
                 cm_model_override: Optional[str] = "claude-3-5-haiku-20240620", # Specific CM model
                 tail_token_limit: int = 30000,
                 **kwargs):
        """
        Initializes the curated memory generator.
        
        Args:
            element: The Element this component is attached to (typically InnerSpace).
            primer_content: The static primer text prepended to LLM contexts.
            agent_model_override: Specific model to use for agent perspective/refinement steps.
            cm_model_override: Specific model to use for the context manager analysis step.
            tail_token_limit: Token limit for the history tail provided by ContextManager.
            **kwargs: Passthrough for BaseComponent.
        """
        super().__init__(element, **kwargs)
        self._state = {
            "primer_content": primer_content,
            "agent_model_override": agent_model_override,
            "cm_model_override": cm_model_override,
            "tail_token_limit": tail_token_limit,
            "last_generation_stats": None
        }
        logger.info(f"CuratedMemoryGenerationComponent initialized for element {element.id if element else 'None'}")

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
        Executes the 6-step memory generation process for a given chunk of messages.

        Args:
            chunk_messages: A list of message dictionaries representing the chunk to process.
                            Assumes these are raw, uncompressed messages.
                            The list should contain at least one message.

        Returns:
            The memory_id of the generated and stored memory, or None on failure.
        """
        start_time = time.time()
        logger.info(f"Starting curated memory generation for chunk of {len(chunk_messages)} messages.")
        if not chunk_messages:
            logger.warning("Cannot generate memory for empty chunk.")
            return None

        ctx_mgr, mem_store, hud = self._get_deps()
        if not all([ctx_mgr, mem_store, hud]):
            return None # Error already logged

        # --- Prepare Context Components --- 
        primer = self._state["primer_content"]
        chunk_start_timestamp = chunk_messages[0].get("timestamp")
        
        # 1. Get History Tail (messages *before* the chunk)
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
            # Simple string formatting for now
            tagged_chunk_str += f"<{role}_turn><to_remember>{json.dumps(content)}</to_remember></{role}_turn>\n"
        
        # 3. Prepare base context (Primer + Tail + Chunk)
        # Note: Formatting needs refinement - how to best represent tail+chunk?
        base_context_str = f"{primer}\n\n--- History Tail ---\n{json.dumps(tail_messages)}\n\n--- Chunk to Remember ---\n{tagged_chunk_str}\n--- End Chunk --- \n\n"

        # --- LLM Calls for Each Step --- 
        llm_provider = hud._llm_provider
        agent_model = self._state.get("agent_model_override") or hud._state.get("model")
        cm_model = self._state.get("cm_model_override")

        try:
            # Step 1: Quote Extraction (Agent Model?)
            logger.debug("Requesting Quote Extraction...")
            quote_context = base_context_str + QUOTE_REQUEST_PROMPT
            quote_response = await llm_provider.complete(
                messages=[{"role": "user", "content": quote_context}], model=agent_model
            )
            quote_content = quote_response.content if quote_response else "(Quote extraction failed)"
            logger.debug(f"Quote Extracted: {quote_content[:100]}...")

            # Step 2: Agent Perspective (Agent Model)
            logger.debug("Requesting Agent Perspective...")
            perspective_context = base_context_str + f"Extracted Quote: {quote_content}\n\n" + AGENT_PERSPECTIVE_PROMPT
            perspective_response = await llm_provider.complete(
                messages=[{"role": "user", "content": perspective_context}], model=agent_model
            )
            perspective_content = perspective_response.content if perspective_response else "(Agent perspective failed)"
            logger.debug(f"Agent Perspective: {perspective_content[:100]}...")

            # Step 3: CM Analysis (CM Model)
            logger.debug("Requesting CM Analysis...")
            analysis_context = base_context_str + f"Extracted Quote: {quote_content}\nAgent Perspective: {perspective_content}\n\n" + CM_ANALYSIS_PROMPT
            analysis_response = await llm_provider.complete(
                messages=[{"role": "user", "content": analysis_context}], model=cm_model
            )
            analysis_content = analysis_response.content if analysis_response else "(CM analysis failed)"
            logger.debug(f"CM Analysis: {analysis_content[:100]}...")
            
            # Step 4/5: Agent Refinement (Agent Model)
            logger.debug("Requesting Agent Refinement...")
            refinement_context = (base_context_str + 
                                  f"Extracted Quote: {quote_content}\n" +
                                  f"Initial Perspective: {perspective_content}\n\n" +
                                  f"<context_manager>Analytical Summary: {analysis_content}</context_manager>\n\n" +
                                  REFINEMENT_REQUEST_PROMPT)
            refinement_response = await llm_provider.complete(
                messages=[{"role": "user", "content": refinement_context}], model=agent_model
            )
            refined_perspective_content = refinement_response.content if refinement_response else "(Agent refinement failed)"
            logger.debug(f"Refined Perspective: {refined_perspective_content[:100]}...")

            # Step 6: Assemble Memory
            # Structure needs finalizing based on requirements doc (3 pairs)
            memory_content = [
                {"role": "system", "content": f"Quote regarding period {chunk_start_timestamp}:"}, # Marker
                {"role": "assistant", "content": quote_content},
                {"role": "system", "content": "Agent perspective on this period:"}, # Marker
                {"role": "assistant", "content": perspective_content},
                {"role": "system", "content": f"<context_manager>Analysis: {analysis_content}</context_manager>"}, # CM Analysis + Refinement request?
                {"role": "assistant", "content": refined_perspective_content} # Final perspective
            ]
            
            memory_metadata = {
                "generation_method": "curated_6_step",
                "agent_model": agent_model,
                "cm_model": cm_model,
                # Add more metadata? Token counts, timings? 
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
                logger.info(f"Successfully generated and stored memory {memory_id} in {(time.time() - start_time):.2f} seconds.")
                self._state["last_generation_stats"] = {"duration": time.time() - start_time, "memory_id": memory_id, "chunk_size": len(chunk_messages)}
                return memory_id
            else:
                 logger.error("Failed to store generated memory.")
                 self._state["last_generation_stats"] = {"duration": time.time() - start_time, "error": "Storage failed", "chunk_size": len(chunk_messages)}
                 return None

        except Exception as e:
            logger.error(f"Error during curated memory generation: {e}", exc_info=True)
            self._state["last_generation_stats"] = {"duration": time.time() - start_time, "error": str(e), "chunk_size": len(chunk_messages)}
            return None

    # TODO: Add methods to trigger memory generation? 
    # e.g., process_history_range(), triggered by event or agent loop? 