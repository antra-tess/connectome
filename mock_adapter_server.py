"""
Mock Socket.IO Adapter Server

Simulates an external adapter process that the ActivityClient connects to.
Listens for outgoing actions and allows triggering incoming events.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

import socketio # Use python-socketio for the server
from aiohttp import web # Use aiohttp for the web server underlying socketio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MockAdapterServer")

# Global Socket.IO server instance
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# Keep track of connected clients (optional)
connected_clients: Dict[str, Dict[str, Any]] = {}

# --- Socket.IO Event Handlers --- 

@sio.event
async def connect(sid, environ, auth):
    client_ip = environ.get('REMOTE_ADDR', 'Unknown')
    logger.info(f"Client connected: sid={sid}, ip={client_ip}, auth={auth}")
    connected_clients[sid] = {"environ": environ, "auth": auth, "connect_time": time.time()}
    # Optionally, send a welcome message or request info?
    # await sio.emit('welcome', {'message': 'Connected to Mock Adapter!'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: sid={sid}")
    if sid in connected_clients:
        del connected_clients[sid]

@sio.event
async def error(sid, data):
    logger.error(f"Socket.IO error from client {sid}: {data}")

# Handler for actions coming FROM the ActivityClient
@sio.on("action_from_core")
async def handle_action_from_core(sid, data):
    """Handles actions sent by the ActivityClient."""
    if not isinstance(data, dict):
        logger.warning(f"Received non-dict action_from_core from {sid}: {data}")
        return
        
    event_type = data.get("event_type")
    action_data = data.get("data", {})
    logger.info(f"Received action from Core (sid={sid}): Type={event_type}, Data={action_data}")
    
    # --- TODO: Add logic here to SIMULATE the effect of the action --- 
    # For example, if event_type == 'send_message', you might store it 
    # or log it specifically.
    # If it was an interaction, you might prepare a simulated response event.
    # -----------------------------------------------------------------
    
    # Optionally acknowledge receipt
    # return {"status": "received", "event_type": event_type}

# --- Function to Simulate Incoming Event --- 

async def simulate_normalized_event(adapter_type="mock", event_type="message_received", data: Optional[Dict[str, Any]] = None):
    """Simulates an event originating from the adapter's service."""
    if not data:
         # Create default message data if none provided
         data = {
            "adapter_name": "mock_adapter_api_1",
            "message_id": f"mock_msg_{int(time.time()*1000)}",
            "conversation_id": "general",
            "sender": {"user_id": "sim_user_456", "display_name": "Simulator"},
            "text": "This is a simulated message!",
            "thread_id": None,
            "attachments": [],
            "timestamp": int(time.time() * 1000)
        }
        
    event_to_send = {
        "adapter_type": adapter_type,
        "event_type": event_type,
        "data": data
    }
    
    logger.info(f"Simulating emission of normalized_event: {event_to_send}")
    # Emit to all connected clients (or specific ones if needed)
    await sio.emit("normalized_event", event_to_send)

# --- Simple HTTP Endpoint for Triggering Simulation --- 

async def trigger_simulation(request):
    """HTTP endpoint to trigger sending a simulated message."""
    try:
        params = await request.post() if request.can_read_body else request.query
        text = params.get("text", "Default simulated message via HTTP")
        conv_id = params.get("conv_id", "http_triggered")
        user_id = params.get("user_id", "http_user")
        user_name = params.get("user_name", "HTTP User")
        
        logger.info(f"HTTP trigger received: text='{text}', conv_id='{conv_id}'")
        
        # Prepare data for the standard simulation function
        sim_data = {
            "adapter_name": "mock_adapter_api_1", # Assuming this ID
            "message_id": f"http_msg_{int(time.time()*1000)}",
            "conversation_id": conv_id,
            "sender": {"user_id": user_id, "display_name": user_name},
            "text": text,
            "thread_id": None,
            "attachments": [],
            "timestamp": int(time.time() * 1000)
        }
        
        # Run the simulation in the background (don't block HTTP response)
        asyncio.create_task(simulate_normalized_event(event_type="message_received", data=sim_data))
        
        return web.Response(text=f"OK. Simulated message event emitted for conversation '{conv_id}'.")
    except Exception as e:
         logger.error(f"Error in HTTP trigger: {e}", exc_info=True)
         return web.Response(text=f"Error: {e}", status=500)

app.router.add_get('/simulate', trigger_simulation) # GET for easy browser testing
app.router.add_post('/simulate', trigger_simulation) # POST is more appropriate

# --- Main Server Execution --- 

async def start_server(host='0.0.0.0', port=5678):
    """Starts the mock adapter server."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    logger.info(f"Starting Mock Adapter Server on http://{host}:{port}")
    await site.start()
    logger.info("Server started. Waiting for connections...")
    # Keep server running until manually stopped
    await asyncio.Event().wait() # Keep running indefinitely

if __name__ == '__main__':
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Mock Adapter Server shutting down.") 