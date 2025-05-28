#!/usr/bin/env python3
"""
Test script for event replay functionality.

Tests the ability to reconstruct system state by replaying timeline events,
verifying that Spaces can restore their state after restart using event replay.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required classes
from elements.elements.space import Space, EventReplayMode
from elements.elements.components.space.timeline_component import TimelineComponent

# Mock space for testing
class TestSpace(Space):
    def __init__(self, space_id="test_space", **kwargs):
        super().__init__(
            element_id=space_id,
            name=f"TestSpace_{space_id}",
            description="Test space for event replay testing",
            **kwargs
        )
        
        # Track state for verification
        self.test_state = {
            'events_received': [],
            'events_replayed': []
        }
        
    def receive_event(self, event_payload, timeline_context):
        """Override to track received events."""
        is_replay = timeline_context.get('replay_mode', False)
        
        if is_replay:
            self.test_state['events_replayed'].append({
                'event_type': event_payload.get('event_type'),
                'event_id': timeline_context.get('original_event_id'),
                'timestamp': timeline_context.get('original_timestamp')
            })
            logger.info(f"[TEST] Replayed event: {event_payload.get('event_type')}")
        else:
            self.test_state['events_received'].append({
                'event_type': event_payload.get('event_type'),
                'timeline_id': timeline_context.get('timeline_id')
            })
            logger.info(f"[TEST] Received event: {event_payload.get('event_type')}")
        
        # Call parent method
        super().receive_event(event_payload, timeline_context)


async def test_event_replay_functionality():
    """Test the complete event replay functionality."""
    logger.info("="*70)
    logger.info("Testing Event Replay Functionality")
    logger.info("="*70)
    
    # Create temporary directory for test storage
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Set environment variables for test
        os.environ['CONNECTOME_STORAGE_TYPE'] = 'file'
        os.environ['CONNECTOME_STORAGE_BASE_DIR'] = str(temp_dir / "test_storage")
        os.environ['CONNECTOME_STORAGE_PRETTY_JSON'] = 'true'
        os.environ['CONNECTOME_EVENT_REPLAY_MODE'] = 'enabled'  # Enable replay
        
        logger.info(f"Using test storage directory: {temp_dir}")
        
        # === Test 1: Create Initial Space and Add Events ===
        logger.info("\n--- Test 1: Creating Initial Space and Adding Events ---")
        
        space1 = TestSpace("replay_test_space_001")
        
        # Wait for async initialization
        await asyncio.sleep(1.0)
        
        # Get factory component for creating test elements
        factory = space1.get_element_factory()
        if not factory:
            logger.error("ElementFactoryComponent not available - cannot test element recreation")
            return 1
        
        # Create some test elements using prefabs (to test recreation)
        test_element_configs = [
            {
                "prefab_name": "basic_agent",  # Assuming this prefab exists
                "element_id": "test_agent_001",
                "element_config": {
                    "name": "TestAgent001",
                    "description": "Test agent for replay testing"
                }
            }
        ]
        
        created_elements = []
        for config in test_element_configs:
            try:
                result = factory.handle_create_element_from_prefab(
                    prefab_name=config["prefab_name"],
                    element_id=config["element_id"],
                    element_config=config["element_config"]
                )
                
                if result.get("success"):
                    created_elements.append(config["element_id"])
                    logger.info(f"✓ Created element: {config['element_id']} from prefab {config['prefab_name']}")
                else:
                    logger.warning(f"Failed to create element from prefab: {result.get('error')}")
            except Exception as e:
                logger.warning(f"Error creating element from prefab {config['prefab_name']}: {e}")
        
        # Add various system events manually
        test_events = [
            {
                "event_type": "component_initialized",
                "data": {
                    "component_id": "comp_001",
                    "component_type": "TestComponent",
                    "owner_element_id": "element_001"
                }
            },
            {
                "event_type": "tool_provider_registered",
                "data": {
                    "tool_name": "test_tool",
                    "provider_id": "provider_001",
                    "tool_description": "A test tool"
                }
            },
            {
                "event_type": "orientation_conversation_set",
                "data": {
                    "compression_engine_id": "engine_001",
                    "orientation_message_count": 3
                }
            }
        ]
        
        # Add events to timeline
        event_ids = []
        for event in test_events:
            event_id = space1.add_event_to_primary_timeline(event)
            event_ids.append(event_id)
            logger.info(f"✓ Added event: {event['event_type']} -> {event_id}")
        
        # Get initial state snapshots
        initial_mounted_elements = space1.get_mounted_elements()
        initial_veil_snapshot = space1.get_flat_veil_snapshot()
        
        logger.info(f"Initial state: {len(initial_mounted_elements)} mounted elements, VEIL cache size: {len(initial_veil_snapshot)}")
        
        # Wait for persistence
        await asyncio.sleep(1.0)
        
        # Verify events were added
        timeline = space1.get_timeline()
        timeline_events = timeline.get_timeline_events(timeline.get_primary_timeline())
        logger.info(f"Total events in timeline: {len(timeline_events)}")
        
        # Shutdown space1
        await space1.get_timeline().shutdown()
        logger.info("✓ Shutdown initial space")
        
        # === Test 2: Create New Space and Test Replay ===
        logger.info("\n--- Test 2: Creating New Space and Testing Replay ---")
        
        space2 = TestSpace("replay_test_space_001")  # Same ID for storage
        
        # Wait for async initialization and replay
        await asyncio.sleep(3.0)  # Allow more time for replay
        
        # Verify replay occurred
        replay_mode = space2.get_replay_mode()
        replayed_count = space2.get_replayed_event_count()
        
        logger.info(f"Replay mode: {replay_mode}")
        logger.info(f"Events replayed: {replayed_count}")
        logger.info(f"Events received during replay: {len(space2.test_state['events_replayed'])}")
        
        # Check timeline events
        timeline2 = space2.get_timeline()
        timeline_events2 = timeline2.get_timeline_events(timeline2.get_primary_timeline())
        logger.info(f"Timeline events loaded: {len(timeline_events2)}")
        
        # === Test 3: Verify Element and VEIL State Recreation ===
        logger.info("\n--- Test 3: Verifying Element and VEIL State Recreation ---")
        
        # Check mounted elements after replay
        replayed_mounted_elements = space2.get_mounted_elements()
        replayed_veil_snapshot = space2.get_flat_veil_snapshot()
        
        logger.info(f"After replay: {len(replayed_mounted_elements)} mounted elements, VEIL cache size: {len(replayed_veil_snapshot)}")
        
        # Verify created elements were recreated
        for element_id in created_elements:
            if element_id in [elem.id for elem in replayed_mounted_elements.values()]:
                logger.info(f"✓ Element {element_id} successfully recreated during replay")
            else:
                logger.warning(f"✗ Element {element_id} was NOT recreated during replay")
        
        # Verify VEIL state consistency
        if len(replayed_veil_snapshot) > 0:
            logger.info("✓ VEIL cache has been regenerated")
            
            # Check if space root exists
            space_root_id = f"{space2.id}_space_root"
            if space_root_id in replayed_veil_snapshot:
                logger.info("✓ Space root VEIL node exists")
            else:
                logger.warning("✗ Space root VEIL node missing")
        else:
            logger.warning("✗ VEIL cache appears empty after replay")
            
        expected_replayable_events = [
            'element_mounted',
            'component_initialized', 
            'tool_provider_registered',
            'orientation_conversation_set'
        ]
        
        replayed_event_types = [event['event_type'] for event in space2.test_state['events_replayed']]
        logger.info(f"Replayed event types: {replayed_event_types}")
        
        # Verify all expected events were replayed
        for expected_event in expected_replayable_events:
            if expected_event in replayed_event_types:
                logger.info(f"✓ Event type {expected_event} was replayed")
            else:
                logger.warning(f"✗ Event type {expected_event} was NOT replayed")
        
        # === Test 4: Test Replay Mode Detection ===
        logger.info("\n--- Test 4: Testing Replay Mode Detection ---")
        
        # Add a new event to space2 (should not be replay)
        new_event = {
            "event_type": "element_mounted",
            "data": {
                "mount_id": "new_element_001", 
                "element_id": "new_element_001",
                "element_name": "NewElement",
                "element_type": "NewElementType",
                "mount_type": "INCLUSION"
            }
        }
        
        # Clear test state
        space2.test_state['events_received'].clear()
        space2.test_state['events_replayed'].clear()
        
        # Add new event
        new_event_id = space2.add_event_to_primary_timeline(new_event)
        await asyncio.sleep(0.5)
        
        # Verify it was received as live event, not replay
        received_events = space2.test_state['events_received']
        replayed_events = space2.test_state['events_replayed']
        
        logger.info(f"New events received (live): {len(received_events)}")
        logger.info(f"New events replayed: {len(replayed_events)}")
        
        if len(received_events) > 0 and len(replayed_events) == 0:
            logger.info("✓ New event correctly processed as live event")
        else:
            logger.warning("✗ New event replay detection failed")
        
        # === Test 5: Test Disabled Replay Mode ===
        logger.info("\n--- Test 5: Testing Disabled Replay Mode ---")
        
        # Shutdown space2
        await space2.get_timeline().shutdown()
        
        # Change environment to disable replay
        os.environ['CONNECTOME_EVENT_REPLAY_MODE'] = 'disabled'
        
        # Create space3 with same ID
        space3 = TestSpace("replay_test_space_001")
        await asyncio.sleep(1.0)
        
        # Verify no replay occurred
        replay_mode3 = space3.get_replay_mode()
        replayed_count3 = space3.get_replayed_event_count()
        
        logger.info(f"Space3 replay mode: {replay_mode3}")
        logger.info(f"Space3 events replayed: {replayed_count3}")
        
        if replay_mode3 == EventReplayMode.DISABLED and replayed_count3 == 0:
            logger.info("✓ Replay correctly disabled")
        else:
            logger.warning("✗ Replay disabling failed")
        
        # Final cleanup
        await space3.get_timeline().shutdown()
        
        logger.info("\n" + "="*70)
        logger.info("✅ EVENT REPLAY FUNCTIONALITY TESTS COMPLETED!")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup test directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up test directory: {temp_dir}")


async def main():
    """Main test function."""
    logger.info("Starting Event Replay Tests")
    return await test_event_replay_functionality()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 