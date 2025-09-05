"""
FIX-style Protocol Implementation for Connectome-Adapter Communication

This protocol ensures reliable, ordered message delivery between Connectome and adapters.
Each side maintains independent sequence counters and can recover from any failure.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Set
from collections import OrderedDict


class MessageDirection(Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"


class SequenceStatus(Enum):
    OK = "ok"                  # Message is next expected sequence
    DUPLICATE = "duplicate"    # Message already processed
    GAP = "gap"               # Missing messages detected
    FUTURE = "future"         # Message too far ahead


@dataclass
class ProtocolMessage:
    """A message with protocol headers"""
    sequence: int
    timestamp: float
    message_type: str
    sender: str
    body: Dict[str, Any]
    requires_ack: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'header': {
                'sequence': self.sequence,
                'timestamp': self.timestamp,
                'message_type': self.message_type,
                'sender': self.sender,
                'requires_ack': self.requires_ack
            },
            'body': self.body
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolMessage':
        header = data.get('header', {})
        return cls(
            sequence=header.get('sequence'),
            timestamp=header.get('timestamp'),
            message_type=header.get('message_type'),
            sender=header.get('sender'),
            body=data.get('body', {}),
            requires_ack=header.get('requires_ack', False)
        )


@dataclass 
class ProtocolState:
    """State for one direction of communication"""
    sequence: int = 0
    last_received: int = 0
    out_of_order_buffer: Dict[int, ProtocolMessage] = field(default_factory=dict)
    pending_acks: Set[int] = field(default_factory=set)


class FIXProtocol:
    """
    FIX-style protocol implementation for reliable message delivery.
    
    This class handles:
    - Independent sequence numbering for each direction
    - Gap detection and retransmission requests
    - Message buffering for out-of-order delivery
    - Message persistence for retransmission
    """
    
    def __init__(self, 
                 node_id: str,
                 send_callback: Callable[[str, Dict[str, Any]], asyncio.Task],
                 process_callback: Callable[[str, Dict[str, Any]], asyncio.Task],
                 storage_ttl: int = 300):
        """
        Initialize the protocol handler.
        
        Args:
            node_id: Unique identifier for this node (e.g., "adapter-discord" or "connectome-main")
            send_callback: Function to send messages over transport (e.g., socketio.emit)
            process_callback: Function to process received messages
            storage_ttl: How long to keep sent messages for retransmission (seconds)
        """
        self.node_id = node_id
        self.send_callback = send_callback
        self.process_callback = process_callback
        self.storage_ttl = storage_ttl
        
        # Independent state for each peer
        self.peer_states: Dict[str, ProtocolState] = {}
        
        # Storage for retransmission
        self.message_storage: OrderedDict[int, ProtocolMessage] = OrderedDict()
        
        # My outbound sequence counter (shared across all peers)
        self.outbound_sequence = 0
        
        # Logger
        self.logger = logging.getLogger(f"FIXProtocol.{node_id}")
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_old_messages())
        
    def get_or_create_peer_state(self, peer_id: str) -> ProtocolState:
        """Get or create state for a peer"""
        if peer_id not in self.peer_states:
            self.peer_states[peer_id] = ProtocolState()
            self.logger.info(f"Created new peer state for {peer_id}")
        return self.peer_states[peer_id]
        
    async def send_message(self, 
                          message_type: str,
                          body: Dict[str, Any],
                          peer_id: Optional[str] = None,
                          requires_ack: bool = False) -> int:
        """
        Send a message with protocol headers.
        
        Args:
            message_type: Type of message (e.g., "bot_request", "bot_response")
            body: Message payload
            peer_id: Target peer (for directed messages)
            requires_ack: Whether explicit acknowledgment is required
            
        Returns:
            Sequence number assigned to the message
        """
        self.outbound_sequence += 1
        
        # Create protocol message
        message = ProtocolMessage(
            sequence=self.outbound_sequence,
            timestamp=time.time(),
            message_type=message_type,
            sender=self.node_id,
            body=body,
            requires_ack=requires_ack
        )
        
        # Store for retransmission
        self.message_storage[self.outbound_sequence] = message
        
        # Send via transport
        await self.send_callback('protocol_message', message.to_dict())
        
        self.logger.debug(f"Sent {message_type} with seq={self.outbound_sequence}")
        
        return self.outbound_sequence
        
    async def handle_incoming_message(self, peer_id: str, data: Dict[str, Any]) -> None:
        """
        Handle an incoming protocol message.
        
        Args:
            peer_id: ID of the sending peer
            data: Raw message data
        """
        try:
            message = ProtocolMessage.from_dict(data)
            peer_state = self.get_or_create_peer_state(peer_id)
            
            # Check sequence status
            status = self._check_sequence_status(message.sequence, peer_state.last_received)
            
            if status == SequenceStatus.OK:
                # Process message
                peer_state.last_received = message.sequence
                await self.process_callback(message.message_type, message.body)
                
                # Process any buffered messages that can now be handled
                await self._process_buffered_messages(peer_id, peer_state)
                
            elif status == SequenceStatus.DUPLICATE:
                self.logger.warning(f"Duplicate message seq={message.sequence} from {peer_id}")
                
            elif status == SequenceStatus.GAP:
                # Request missing messages
                gap_start = peer_state.last_received + 1
                gap_end = message.sequence - 1
                
                self.logger.warning(f"Gap detected from {peer_id}: missing seq {gap_start}-{gap_end}")
                
                # Buffer this message
                peer_state.out_of_order_buffer[message.sequence] = message
                
                # Request retransmission
                await self.send_callback('resend_request', {
                    'from_sequence': gap_start,
                    'to_sequence': gap_end,
                    'requester': self.node_id
                })
                
        except Exception as e:
            self.logger.error(f"Error handling message from {peer_id}: {e}", exc_info=True)
            
    async def handle_resend_request(self, peer_id: str, data: Dict[str, Any]) -> None:
        """
        Handle a request to resend messages.
        
        Args:
            peer_id: ID of the requesting peer
            data: Request data with from_sequence and to_sequence
        """
        from_seq = data.get('from_sequence')
        to_seq = data.get('to_sequence')
        
        self.logger.info(f"Resend request from {peer_id}: seq {from_seq}-{to_seq}")
        
        for seq in range(from_seq, to_seq + 1):
            if seq in self.message_storage:
                # Resend the message
                await self.send_callback('protocol_message', self.message_storage[seq].to_dict())
                self.logger.debug(f"Resent seq={seq} to {peer_id}")
            else:
                self.logger.error(f"Cannot resend seq={seq} - not in storage")
                
    async def handle_sequence_sync(self, peer_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle sequence synchronization on connection.
        
        Args:
            peer_id: ID of the connecting peer
            data: Sync data from peer
            
        Returns:
            Our sync data for the peer
        """
        peer_outbound = data.get('my_outbound_seq', 0)
        peer_expects = data.get('expecting_inbound_seq', 1)
        
        peer_state = self.get_or_create_peer_state(peer_id)
        
        # Check if we need to request missing messages
        if peer_outbound > peer_state.last_received:
            # We're missing messages
            await self.send_callback('resend_request', {
                'from_sequence': peer_state.last_received + 1,
                'to_sequence': peer_outbound,
                'requester': self.node_id
            })
            
        # Return our state
        return {
            'my_outbound_seq': self.outbound_sequence,
            'expecting_inbound_seq': peer_state.last_received + 1
        }
        
    def _check_sequence_status(self, received_seq: int, last_received: int) -> SequenceStatus:
        """Check if a sequence number is valid"""
        expected = last_received + 1
        
        if received_seq == expected:
            return SequenceStatus.OK
        elif received_seq <= last_received:
            return SequenceStatus.DUPLICATE
        elif received_seq == expected + 1:
            return SequenceStatus.GAP
        else:
            return SequenceStatus.FUTURE
            
    async def _process_buffered_messages(self, peer_id: str, peer_state: ProtocolState) -> None:
        """Process any buffered messages that are now in sequence"""
        while True:
            next_seq = peer_state.last_received + 1
            
            if next_seq in peer_state.out_of_order_buffer:
                # This message can now be processed
                message = peer_state.out_of_order_buffer.pop(next_seq)
                peer_state.last_received = next_seq
                
                await self.process_callback(message.message_type, message.body)
                self.logger.debug(f"Processed buffered message seq={next_seq}")
            else:
                # No more consecutive messages to process
                break
                
    async def _cleanup_old_messages(self) -> None:
        """Periodically clean up old messages from storage"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                sequences_to_remove = []
                
                for seq, message in self.message_storage.items():
                    if current_time - message.timestamp > self.storage_ttl:
                        sequences_to_remove.append(seq)
                        
                for seq in sequences_to_remove:
                    del self.message_storage[seq]
                    
                if sequences_to_remove:
                    self.logger.debug(f"Cleaned up {len(sequences_to_remove)} old messages")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                
    async def shutdown(self) -> None:
        """Clean shutdown of the protocol"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass 