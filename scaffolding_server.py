#!/usr/bin/env python3
"""
Scaffolding Web Server

A Flask web server that provides a manual interface for LLM testing.
This server receives LLM contexts from the ScaffoldingLLMProvider and allows
manual response input through a web interface.

Can be also used with LiteLLMProvider in Observer mode. In this case, the server
will receive the context from the LiteLLMProvider and LLM responses. There will be
no possibility to submit a manual response.

Usage:
    - With ScaffoldingLLMProvider:
        python scaffolding_server.py
    - With LiteLLMProvider in Observer mode:
        OBSERVER=true python scaffolding_server.py

Then configure the agent with (only for ScaffoldingLLMProvider):
    CONNECTOME_LLM_TYPE=scaffolding
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state for managing context and responses
current_context: Optional[Dict[str, Any]] = None
pending_response: Optional[str] = None
response_ready = threading.Event()
context_history: List[Dict[str, Any]] = []

# Lock for thread safety
state_lock = threading.Lock()

# Keep track of Observer Mode
observer = False

@app.route('/')
def index():
    """Redirect to the test interface."""
    global observer
    return render_template('scaffolding_interface.html', observer=observer)

@app.route('/test-interface')
def test_interface():
    """Serve the main testing interface."""
    global observer
    return render_template('scaffolding_interface.html', observer=observer)

@app.route('/submit-context', methods=['POST'])
def submit_context():
    """
    Receive context from ScaffoldingLLMProvider and wait for manual response.

    This endpoint:
    1. Receives the context data from the LLM provider
    2. Stores it globally for the web interface to display
    3. Waits for a manual response to be submitted
    4. Returns the manual response to the LLM provider
    """
    global current_context, pending_response, context_history

    try:
        context_data = request.json

        with state_lock:
            current_context = context_data
            pending_response = None
            response_ready.clear()

            # Add to history
            context_history.append({
                **context_data,
                "received_at": datetime.now().isoformat(),
                "status": "waiting_for_response"
            })

            # Keep only last 10 entries in history
            if len(context_history) > 10:
                context_history = context_history[-10:]

        logger.info(f"Received context with {len(context_data.get('messages', []))} messages")

        # Wait for manual response (with timeout)
        timeout = 300  # 5 minutes
        if response_ready.wait(timeout=timeout):
            with state_lock:
                manual_response = pending_response or "No response provided"
                # Update history status
                if context_history:
                    context_history[-1]["status"] = "completed"
                    context_history[-1]["response"] = manual_response
                    context_history[-1]["responded_at"] = datetime.now().isoformat()
        else:
            manual_response = "Error: Timeout waiting for manual response"
            logger.warning(f"Timeout waiting for manual response after {timeout}s")
            with state_lock:
                if context_history:
                    context_history[-1]["status"] = "timeout"

        return jsonify({
            "status": "success",
            "manual_response": manual_response
        })

    except Exception as e:
        logger.error(f"Error in submit_context: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "manual_response": f"Error: {str(e)}"
        }), 500

@app.route('/get-current-context')
def get_current_context():
    """Get the current context for display in the web interface."""
    with state_lock:
        if current_context:
            return jsonify({
                "status": "success",
                "context": current_context,
                "has_context": True
            })
        else:
            return jsonify({
                "status": "success",
                "context": None,
                "has_context": False
            })

@app.route('/submit-response', methods=['POST'])
def submit_response():
    """Receive manual response from the web interface."""
    global pending_response

    try:
        data = request.json
        manual_response = data.get('response', '').strip()

        if not manual_response:
            return jsonify({
                "status": "error",
                "message": "Response cannot be empty"
            }), 400

        with state_lock:
            pending_response = manual_response
            response_ready.set()

        logger.info(f"Received manual response: {len(manual_response)} characters")

        return jsonify({
            "status": "success",
            "message": "Response submitted successfully"
        })

    except Exception as e:
        logger.error(f"Error in submit_response: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/get-history')
def get_history():
    """Get the history of context/response exchanges."""
    with state_lock:
        return jsonify({
            "status": "success",
            "history": context_history
        })

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear the context/response history."""
    global context_history

    with state_lock:
        context_history.clear()

    return jsonify({
        "status": "success",
        "message": "History cleared"
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_context": current_context is not None,
        "history_count": len(context_history)
    })

# Ensure templates directory exists
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
    logger.info(f"Created templates directory: {templates_dir}")

def main():
    """Start the scaffolding web server."""
    print("\n" + "="*60)
    print("🔧 LLM Testing Scaffolding Server")
    print("="*60)
    print(f"📍 Web Interface: http://localhost:6200")
    print(f"📁 Templates Directory: {templates_dir}")
    print("\n🌐 Interface Features:")
    print("   • View agent context in real-time")
    print("   • Manually provide LLM responses")
    print("   • Session history and replay")
    print("   • Zero LLM API costs")
    print("="*60 + "\n")

    try:
        global observer

        observer = os.getenv('OBSERVER', 'false').lower() == 'true'
        app.run(host='0.0.0.0', port=6200, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Scaffolding server stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)

if __name__ == '__main__':
    main()