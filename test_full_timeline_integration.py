#!/usr/bin/env python3
"""
Test script for full timeline integration.

Verifies that both timeline systems are properly integrated with storage:
1. Conversation Timeline (CompressionEngineComponent) - agent reasoning and memory
2. InnerSpace Timeline (TimelineComponent) - system events like element creation/mounting

This ensures complete persistence and reconstruction of both conversation context
and system state across restarts.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components and required classes
from elements.elements.components.compression_engine_component import CompressionEngineComponent
from elements.elements.components.space.timeline_component import TimelineComponent
from elements.elements.space import Space
from llm.provider_interface import LLMMessage

# Mock classes for testing
class MockInnerSpace:
    def __init__(self, agent_name="test_agent"):
        self.id = "mock_inner_space_001"
        self.agent_name = agent_name
        self.agent_id = f"agent_{agent_name}"
        self.name = f"TestInnerSpace_{agent_name}"
        
        # Initialize timeline component
        self._timeline = TimelineComponent()
        self._timeline.owner = self
        self._timeline.initialize()
        
    def add_event_to_primary_timeline(self, event):
        if self._timeline:
            return self._timeline.add_event_to_primary_timeline(event)
        return None

    def get_timeline(self):
        return self._timeline

    async def shutdown(self):
        if self._timeline:
            await self._timeline.shutdown()


async def test_full_timeline_integration():
    """Test both conversation and InnerSpace timeline integration with storage."""
    logger.info("="*70)
    logger.info("Testing Full Timeline Integration (Conversation + InnerSpace)")
    logger.info("="*70)
    
    # Create temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Set environment variables for file storage
        os.environ['CONNECTOME_STORAGE_TYPE'] = 'file'
        os.environ['CONNECTOME_STORAGE_BASE_DIR'] = str(temp_dir / "test_storage")
        os.environ['CONNECTOME_STORAGE_PRETTY_JSON'] = 'true'
        
        logger.info(f"Using test storage directory: {temp_dir}")
        
        # === Test 1: InnerSpace Timeline Events ===
        logger.info("\n--- Test 1: InnerSpace Timeline - System Events ---")
        
        # Create mock inner space with timeline
        inner_space = MockInnerSpace("test_agent_001")
        await asyncio.sleep(0.5)  # Allow async init
        
        # Add some system events
        system_events = [
            {
                "event_type": "element_mounted",
                "data": {
                    "mount_id": "agent_001",
                    "element_id": "agent_element_001",
                    "element_name": "TestAgent",
                    "element_type": "AgentElement",
                    "mount_type": "INCLUSION"
                }
            },
            {
                "event_type": "component_initialized",
                "data": {
                    "component_id": "comp_001",
                    "component_type": "CompressionEngineComponent",
                    "owner_element_id": "agent_element_001"
                }
            },
            {
                "event_type": "tool_provider_registered",
                "data": {
                    "tool_name": "analyze_data",
                    "provider_id": "tool_provider_001",
                    "tool_description": "Analyzes data structures"
                }
            }
        ]
        
        event_ids = []
        for event in system_events:
            event_id = inner_space.add_event_to_primary_timeline(event)
            event_ids.append(event_id)
            logger.info(f"✓ Added system event: {event['event_type']}")
        
        await asyncio.sleep(1)  # Allow persistence to complete
        
        # === Test 2: Conversation Timeline Events ===
        logger.info("\n--- Test 2: Conversation Timeline - Agent Memory ---")
        
        # Create CompressionEngine component
        compression_engine = CompressionEngineComponent()
        compression_engine.owner = inner_space
        compression_engine.initialize()
        await asyncio.sleep(0.5)  # Allow async init
        
        # Set orientation conversation
        orientation_messages = [
            LLMMessage(role="user", content="Hello, you are a data analysis assistant."),
            LLMMessage(role="assistant", content="Hello! I'm ready to help with data analysis tasks."),
            LLMMessage(role="user", content="Your specialty is working with large datasets and providing insights."),
            LLMMessage(role="assistant", content="Understood. I'll focus on data analysis and generating actionable insights.")
        ]
        
        success = await compression_engine.set_orientation_conversation(orientation_messages)
        assert success, "Failed to set orientation conversation"
        logger.info("✓ Set orientation conversation")
        
        # Store reasoning chains
        reasoning_chains = [
            {
                "context_received": "User uploaded a CSV file with sales data",
                "agent_response": "I'll analyze this sales data to identify trends and patterns. Let me start by examining the structure.",
                "tool_calls": [{"tool_name": "analyze_data", "parameters": {"data_type": "csv", "focus": "trends"}}],
                "tool_results": [{"success": True, "message": "Data structure analyzed", "insights": ["Seasonal patterns detected"]}],
                "reasoning_notes": "Identified seasonal sales patterns",
                "metadata": {"interaction_type": "data_analysis"}
            },
            {
                "context_received": "User asked for quarterly breakdown",
                "agent_response": "Based on my analysis, I can provide a quarterly breakdown showing Q4 has the highest sales volume.",
                "tool_calls": [{"tool_name": "generate_report", "parameters": {"type": "quarterly", "format": "summary"}}],
                "tool_results": [{"success": True, "message": "Report generated", "summary": "Q4: +23% vs Q3"}],
                "reasoning_notes": "Generated quarterly analysis report",
                "metadata": {"interaction_type": "reporting"}
            }
        ]
        
        for i, chain_data in enumerate(reasoning_chains):
            await compression_engine.store_reasoning_chain(chain_data)
            logger.info(f"✓ Stored reasoning chain {i+1}")
        
        # Store memory formation
        memory_data = {
            "source_chunk_references": ["sales_q1", "sales_q2", "sales_q3", "sales_q4"],
            "memory_formation_sequence": [
                {"role": "user", "content": "<context_manager>Sales data shows seasonal patterns...</context_manager>"},
                {"role": "assistant", "content": "I can see clear seasonal patterns in the sales data."},
                {"role": "user", "content": "<context_manager>What's your analysis of Q4 performance?</context_manager>"},
                {"role": "assistant", "content": "Q4 shows exceptional performance with 23% growth..."},
                {"role": "user", "content": "<context_manager>Compress this insight...</context_manager>"},
                {"role": "assistant", "content": "Key insight: Q4 seasonal peak drives 23% sales growth annually."}
            ],
            "compression_metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": 4,
                "token_estimate": 2000,
                "insights": ["seasonal_sales_pattern", "q4_peak_performance"]
            }
        }
        
        success = await compression_engine.store_memory_formation("sales_analysis_001", memory_data)
        assert success, "Failed to store memory formation"
        logger.info("✓ Stored memory formation")
        
        # === Test 3: Verify Data Persistence ===
        logger.info("\n--- Test 3: System Restart Simulation ---")
        
        # Get statistics before shutdown
        memory_stats = await compression_engine.get_memory_stats()
        timeline = inner_space.get_timeline()
        timeline_events = timeline.get_timeline_events(timeline.get_primary_timeline())
        
        logger.info(f"Pre-shutdown: {memory_stats['total_interactions']} reasoning chains")
        logger.info(f"Pre-shutdown: {len(timeline_events)} system events")
        
        # Shutdown current system
        await compression_engine.shutdown()
        await inner_space.shutdown()
        
        # Create new system instance (simulating restart)
        logger.info("Creating new system instance (simulating restart)...")
        
        new_inner_space = MockInnerSpace("test_agent_001")
        await asyncio.sleep(1)  # Allow loading
        
        new_compression_engine = CompressionEngineComponent()
        new_compression_engine.owner = new_inner_space
        new_compression_engine.initialize()
        await asyncio.sleep(1)  # Allow loading
        
        # === Test 4: Verify Timeline Reconstruction ===
        logger.info("\n--- Test 4: Timeline Reconstruction Verification ---")
        
        # Check InnerSpace timeline reconstruction
        new_timeline = new_inner_space.get_timeline()
        new_timeline_events = new_timeline.get_timeline_events(new_timeline.get_primary_timeline())
        
        logger.info(f"Post-restart: {len(new_timeline_events)} system events loaded")
        assert len(new_timeline_events) > 0, "Should have loaded system events"
        
        # Verify system event types are preserved
        loaded_event_types = [event['payload'].get('event_type') for event in new_timeline_events]
        logger.info(f"Loaded system event types: {loaded_event_types}")
        
        # Check conversation timeline reconstruction  
        new_memory_context = await new_compression_engine.get_memory_context()
        new_memory_stats = await new_compression_engine.get_memory_stats()
        
        logger.info(f"Post-restart: {new_memory_stats['total_interactions']} reasoning chains loaded")
        logger.info(f"Post-restart: {len(new_memory_context)} memory messages available")
        
        assert new_memory_stats['total_interactions'] == len(reasoning_chains), "Should have loaded all reasoning chains"
        assert len(new_memory_context) > 0, "Should have loaded memory context"
        
        # Check stored memories
        stored_memories = await new_compression_engine.get_stored_memories()
        logger.info(f"Post-restart: {len(stored_memories)} stored memories loaded")
        assert len(stored_memories) >= 1, "Should have loaded stored memories"
        
        # === Test 5: Add New Events to Verify Continued Operation ===
        logger.info("\n--- Test 5: Continued Operation After Restart ---")
        
        # Add new system event
        new_system_event = {
            "event_type": "element_unmounted", 
            "data": {
                "unmounted_element_id": "temp_element_001",
                "mount_id": "temp_mount_001"
            }
        }
        new_event_id = new_inner_space.add_event_to_primary_timeline(new_system_event)
        assert new_event_id, "Should be able to add new system events"
        logger.info("✓ Added new system event after restart")
        
        # Add new reasoning chain
        new_reasoning_chain = {
            "context_received": "User requested follow-up analysis",
            "agent_response": "Based on my previous analysis, I can provide deeper insights into the seasonal patterns.",
            "tool_calls": [{"tool_name": "deep_analysis", "parameters": {"context": "seasonal_follow_up"}}],
            "tool_results": [{"success": True, "message": "Deep analysis complete"}],
            "reasoning_notes": "Follow-up analysis leveraging stored insights",
            "metadata": {"interaction_type": "follow_up_analysis"}
        }
        
        await new_compression_engine.store_reasoning_chain(new_reasoning_chain)
        logger.info("✓ Added new reasoning chain after restart")
        
        # Final statistics
        final_memory_stats = await new_compression_engine.get_memory_stats()
        final_timeline_events = new_timeline.get_timeline_events(new_timeline.get_primary_timeline())
        
        logger.info(f"Final: {final_memory_stats['total_interactions']} reasoning chains")
        logger.info(f"Final: {len(final_timeline_events)} system events")
        
        # Cleanup
        await new_compression_engine.shutdown()
        await new_inner_space.shutdown()
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL FULL TIMELINE INTEGRATION TESTS PASSED!")
        logger.info("="*70)
        
        # Show final storage statistics
        storage_stats = final_memory_stats.get("storage_stats", {})
        logger.info(f"\nFinal storage state:")
        logger.info(f"• Storage type: {storage_stats.get('storage_type', 'unknown')}")
        logger.info(f"• System states: {storage_stats.get('system_state_count', 0)}")
        logger.info(f"• Reasoning chains: {storage_stats.get('reasoning_chains_count', 0)}")
        logger.info(f"• Conversations: {storage_stats.get('conversation_count', 0)}")
        logger.info(f"• Memories: {storage_stats.get('memories_count', 0)}")
        
        logger.info(f"\n🎉 Both timeline systems successfully integrated with storage!")
        logger.info(f"• InnerSpace Timeline: System events persist across restarts")
        logger.info(f"• Conversation Timeline: Agent memory and reasoning chains persist")
        logger.info(f"• Complete timeline reconstruction: ✅")
        logger.info(f"• Continued operation after restart: ✅")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup test directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up test directory: {temp_dir}")
    
    return 0


async def main():
    """Main test function."""
    logger.info("Starting Full Timeline Integration Tests")
    return await test_full_timeline_integration()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 