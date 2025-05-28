#!/usr/bin/env python3
"""
Test script for CompressionEngine storage integration.

Verifies that the CompressionEngineComponent properly stores and loads
conversation data, reasoning chains, and memories using the pluggable storage system.
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

# Import the CompressionEngine and required classes
from elements.elements.components.compression_engine_component import CompressionEngineComponent
from llm.provider_interface import LLMMessage

# Mock Element class for testing
class MockElement:
    def __init__(self, agent_name="test_agent"):
        self.id = "mock_element_001"
        self.agent_name = agent_name
        self.agent_id = f"agent_{agent_name}"
        self.name = f"TestAgent_{agent_name}"
        
    def add_event_to_primary_timeline(self, event):
        logger.info(f"Timeline event: {event['event_type']} - {event['data']}")


async def test_compression_engine_storage():
    """Test CompressionEngine storage integration."""
    logger.info("="*60)
    logger.info("Testing CompressionEngine Storage Integration")
    logger.info("="*60)
    
    # Create temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Set environment variables for file storage
        os.environ['CONNECTOME_STORAGE_TYPE'] = 'file'
        os.environ['CONNECTOME_STORAGE_BASE_DIR'] = str(temp_dir / "test_storage")
        os.environ['CONNECTOME_STORAGE_PRETTY_JSON'] = 'true'
        
        logger.info(f"Using test storage directory: {temp_dir}")
        
        # Create CompressionEngine component
        mock_element = MockElement("test_agent_001")
        compression_engine = CompressionEngineComponent()
        compression_engine.owner = mock_element
        
        # Initialize the component
        logger.info("Initializing CompressionEngine...")
        success = compression_engine.initialize()
        assert success, "CompressionEngine initialization failed"
        
        # Wait a moment for async initialization to complete
        await asyncio.sleep(1)
        
        # Test 1: Set orientation conversation
        logger.info("\n--- Test 1: Setting orientation conversation ---")
        orientation_messages = [
            LLMMessage(role="user", content="Hello, I'm setting up your orientation."),
            LLMMessage(role="assistant", content="Thank you! I'm ready to learn my role."),
            LLMMessage(role="user", content="You are a helpful AI assistant focused on storage systems."),
            LLMMessage(role="assistant", content="Understood. I'll help with storage-related tasks.")
        ]
        
        success = await compression_engine.set_orientation_conversation(orientation_messages)
        assert success, "Failed to set orientation conversation"
        logger.info("✓ Orientation conversation set successfully")
        
        # Test 2: Store reasoning chains
        logger.info("\n--- Test 2: Storing reasoning chains ---")
        reasoning_chains = [
            {
                "context_received": "User asked about storage backends",
                "agent_response": "I recommend starting with file storage for development and upgrading to SQLite for production.",
                "tool_calls": [{"tool_name": "analyze_storage", "parameters": {"type": "comparison"}}],
                "tool_results": [{"success": True, "message": "Analysis complete"}],
                "reasoning_notes": "Provided storage recommendations",
                "metadata": {"interaction_type": "advisory"}
            },
            {
                "context_received": "User needs help with data migration",
                "agent_response": "I can help you migrate from file storage to SQLite. Let me check the migration tools.",
                "tool_calls": [{"tool_name": "check_migration_tools", "parameters": {}}],
                "tool_results": [{"success": True, "message": "Migration tools available"}],
                "reasoning_notes": "Helping with storage migration",
                "metadata": {"interaction_type": "technical_support"}
            }
        ]
        
        for i, chain_data in enumerate(reasoning_chains):
            await compression_engine.store_reasoning_chain(chain_data)
            logger.info(f"✓ Stored reasoning chain {i+1}")
        
        # Test 3: Get memory context
        logger.info("\n--- Test 3: Retrieving memory context ---")
        memory_context = await compression_engine.get_memory_context()
        logger.info(f"Retrieved {len(memory_context)} memory messages")
        
        # Should include orientation + reasoning chains
        expected_messages = len(orientation_messages) + (len(reasoning_chains) * 2)  # Each chain has user + assistant
        assert len(memory_context) >= expected_messages, f"Expected at least {expected_messages} messages, got {len(memory_context)}"
        logger.info("✓ Memory context retrieved successfully")
        
        # Test 4: Store conversation snapshot
        logger.info("\n--- Test 4: Storing conversation snapshot ---")
        conversation_snapshot = {
            "data": {
                "chats": [{
                    "id": "test_conversation_001",
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [
                        {"role": "user", "content": "Test message 1", "createdAt": datetime.now(timezone.utc).isoformat()},
                        {"role": "assistant", "content": "Test response 1", "createdAt": datetime.now(timezone.utc).isoformat()}
                    ]
                }]
            }
        }
        
        success = await compression_engine.store_conversation_snapshot(conversation_snapshot)
        assert success, "Failed to store conversation snapshot"
        logger.info("✓ Conversation snapshot stored successfully")
        
        # Test 5: Store memory formation
        logger.info("\n--- Test 5: Storing memory formation ---")
        memory_data = {
            "source_chunk_references": ["chunk_1", "chunk_2"],
            "memory_formation_sequence": [
                {"role": "user", "content": "<context_manager>Quote from chunk...</context_manager>"},
                {"role": "assistant", "content": "Here's the relevant quote..."},
                {"role": "user", "content": "<context_manager>What's your perspective?</context_manager>"},
                {"role": "assistant", "content": "My perspective on this..."},
                {"role": "user", "content": "<context_manager>Analytical summary...</context_manager>"},
                {"role": "assistant", "content": "Refined analysis..."}
            ],
            "compression_metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": 2,
                "token_estimate": 1500
            }
        }
        
        success = await compression_engine.store_memory_formation("memory_001", memory_data)
        assert success, "Failed to store memory formation"
        logger.info("✓ Memory formation stored successfully")
        
        # Test 6: Get memory statistics
        logger.info("\n--- Test 6: Getting memory statistics ---")
        stats = await compression_engine.get_memory_stats()
        logger.info(f"Memory stats: {stats}")
        
        assert stats["total_interactions"] == len(reasoning_chains), f"Expected {len(reasoning_chains)} interactions"
        assert stats["storage_initialized"] == True, "Storage should be initialized"
        assert stats["agent_name"] == "test_agent_001", "Agent name should match"
        logger.info("✓ Memory statistics retrieved successfully")
        
        # Test 7: Create new CompressionEngine and verify data persistence
        logger.info("\n--- Test 7: Testing data persistence ---")
        
        # Shutdown current engine
        await compression_engine.shutdown()
        
        # Create new CompressionEngine with same agent
        new_mock_element = MockElement("test_agent_001")
        new_compression_engine = CompressionEngineComponent()
        new_compression_engine.owner = new_mock_element
        
        # Initialize and wait for async loading
        success = new_compression_engine.initialize()
        assert success, "New CompressionEngine initialization failed"
        await asyncio.sleep(1)  # Wait for async loading
        
        # Check if data was loaded
        new_memory_context = await new_compression_engine.get_memory_context()
        logger.info(f"New engine loaded {len(new_memory_context)} memory messages")
        
        # Should have loaded the stored reasoning chains
        assert len(new_memory_context) > 0, "Should have loaded existing memory context"
        logger.info("✓ Data persistence verified successfully")
        
        # Test 8: Load stored memories
        logger.info("\n--- Test 8: Loading stored memories ---")
        stored_memories = await new_compression_engine.get_stored_memories()
        logger.info(f"Loaded {len(stored_memories)} stored memories")
        
        assert len(stored_memories) >= 1, "Should have at least one stored memory"
        logger.info("✓ Stored memories loaded successfully")
        
        # Cleanup
        await new_compression_engine.shutdown()
        
        logger.info("\n" + "="*60)
        logger.info("✓ ALL COMPRESSION ENGINE STORAGE TESTS PASSED!")
        logger.info("="*60)
        
        # Show final storage state
        storage_stats = stats.get("storage_stats", {})
        logger.info(f"\nFinal storage state:")
        logger.info(f"• Storage type: {storage_stats.get('storage_type', 'unknown')}")
        logger.info(f"• Conversations: {storage_stats.get('conversation_count', 0)}")
        logger.info(f"• Reasoning chains: {storage_stats.get('reasoning_chains_count', 0)}")
        logger.info(f"• Memories: {storage_stats.get('memories_count', 0)}")
        logger.info(f"• System states: {storage_stats.get('system_state_count', 0)}")
        
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
    logger.info("Starting CompressionEngine Storage Integration Tests")
    return await test_compression_engine_storage()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 