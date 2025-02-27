#!/usr/bin/env python3
"""
Bot Framework Main Entry Point
Initializes the Socket.IO server and sets up message handling.
"""

import socketio
import eventlet
from config import SOCKET_HOST, SOCKET_PORT
from messaging.listener import register_socket_events
from utils.llm import initialize_litellm
from agent.agent import Agent

# Initialize agent
agent = Agent()

# Initialize Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

if __name__ == '__main__':
    # Initialize LiteLLM
    initialize_litellm()
    
    # Register Socket.IO event handlers
    register_socket_events(sio, agent)
    
    # Start server
    print(f"Starting Bot Framework server on {SOCKET_HOST}:{SOCKET_PORT}")
    eventlet.wsgi.server(eventlet.listen((SOCKET_HOST, SOCKET_PORT)), app) 