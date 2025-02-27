# Bot Framework

A modular architecture for an open-source system that enables a single LLM-based agent to participate in multi-user conversations across different platforms while utilizing various tools.

## Core Features

- **Multi-Platform Support**: Connect to multiple messaging platforms through platform-specific normalizing layers
- **Pluggable Tool-Usage Protocols**: Support for ReAct, Function Calling, and custom protocols
- **Tool Registry**: Easy registration and execution of tools via a decorator pattern
- **Context Management**: Intelligent handling of conversation context for the agent
- **Socket.IO Communication**: Lightweight message passing between the agent and normalizing layers

## Architecture Overview

The system is designed with these major components:

1. **System Prompt Engineering**: Configurable prompts with pluggable tool-usage protocols
2. **Message Handling**: Socket.IO listener for receiving messages from normalizing layers
3. **Context Management**: Organizes messages in proper sequence for the agent
4. **Tool Infrastructure**: Extensible tool registry and execution management
5. **Agent Capabilities**: Multi-user chat, document editing, memory management, etc.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bot_framework.git
   cd bot_framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:
   ```
   # Socket.IO Server
   SOCKET_HOST=0.0.0.0
   SOCKET_PORT=5000
   
   # LiteLLM Settings
   LLM_API_KEY=your_api_key_here
   LLM_MODEL=gpt-4
   LLM_PROVIDER=openai
   
   # Agent Settings
   DEFAULT_PROTOCOL=react
   ```

## Usage

### Starting the Server

```bash
python main.py
```

### Creating Custom Tools

Create tools by using the `register_tool` decorator:

```python
from tools.registry import register_tool

@register_tool(
    name="send_email",
    description="Send an email to a specified recipient",
    parameter_descriptions={
        "recipient": "Email address of the recipient",
        "subject": "Subject line of the email",
        "body": "Content of the email"
    }
)
def send_email(recipient, subject, body):
    # Implementation details here
    return f"Email sent to {recipient}"
```

### Implementing a Custom Protocol

Extend the `BaseProtocol` class to create a custom tool-usage protocol:

```python
from agent.protocol.base_protocol import BaseProtocol

class MyCustomProtocol(BaseProtocol):
    def format_system_prompt(self, base_prompt, tools):
        # Your implementation here
        pass
    
    def extract_tool_calls(self, llm_response):
        # Your implementation here
        pass
    
    def format_tool_result(self, tool_name, result):
        # Your implementation here
        pass
    
    def extract_final_response(self, llm_response):
        # Your implementation here
        pass
    
    def format_for_litellm(self, base_prompt, messages, tools):
        # Your implementation here
        pass
```

## Normalizing Layers

The Bot Framework communicates with platform-specific normalizing layers via Socket.IO. Each normalizing layer is responsible for:

1. Converting platform-specific message formats to a standard format
2. Sending normalized messages to the Bot Framework
3. Receiving responses from the Bot Framework and delivering them to the platform

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 