"""
Mock Adapter Server

Simulates external adapters listening for action requests from the ActivityClient.
Listens on a Socket.IO connection for 'action_request' events and logs them.
"""

import asyncio
import logging
import socketio
from aiohttp import web

# --- Configuration ---
HOST = 'localhost' # Listen only on localhost
PORT = 8081        # Port for the mock server
LOG_LEVEL = logging.INFO

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MockAdapterServer")

# --- Socket.IO Server Setup ---
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# --- Socket.IO Event Handlers ---

@sio.event
def connect(sid, environ):
    """Handle new client connections."""
    logger.info(f"Client connected: {sid} (from {environ.get('REMOTE_ADDR', 'unknown')})")
    # You could potentially store connected clients if needed later

@sio.event
def disconnect(sid):
    """Handle client disconnections."""
    logger.info(f"Client disconnected: {sid}")

@sio.on('action_request')
async def handle_action_request(sid, data):
    """Listen for and log action requests from the ActivityClient."""
    logger.info(f"Received action_request from {sid}:")
    
    # Basic validation
    if not isinstance(data, dict):
        logger.warning(f"  Invalid action_request format (not a dict): {data}")
        return
        
    adapter_id = data.get('adapter_id')
    action_type = data.get('action_type')
    payload = data.get('payload', {})
    request_id = data.get('request_id') # Important for tracking
    
    if not all([adapter_id, action_type, request_id]):
        logger.warning(f"  Malformed action_request (missing fields): {data}")
        return

    logger.info(f"  Request ID: {request_id}")
    logger.info(f"  Adapter ID: {adapter_id}")
    logger.info(f"  Action Type: {action_type}")
    logger.info(f"  Payload: {payload}")

    # --- Future Extension Point ---
    # Here you could:
    # 1. Simulate success/failure responses back to the ActivityClient.
    # 2. Simulate incoming events (e.g., a new message arriving) by emitting
    #    events back to the ActivityClient (would require client-side handlers).
    # 3. Store requests for later inspection or replay.
    
    # For now, we just log it.
    # We could send a simple acknowledgement back if needed:
    # await sio.emit('action_ack', {'request_id': request_id, 'status': 'received'}, room=sid)

# --- Web Server Setup ---
async def main():
    """Main entry point to start the server."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    logger.info(f"Starting Mock Adapter Server on http://{HOST}:{PORT}")
    await site.start()
    logger.info("Server started. Waiting for connections...")
    # Keep the server running indefinitely
    await asyncio.Event().wait()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down.")
