# Bot Framework

A modular, extensible framework for building conversational AI agents with environment-based architecture.

## Architecture Overview

The Bot Framework is built around an environment-based architecture that provides a flexible and extensible way to organize capabilities and tools. The core components include:

### Environment System

The environment system is the foundation of the Bot Framework, allowing for a hierarchical organization of capabilities:

- **Environment Manager**: Orchestrates all environments, keeps track of registered environments, and facilitates communication between them.
- **Interface Layer**: Provides a unified API for agents to interact with the environment system, abstracting away the details of environment mounting and tool execution.
- **System Environment**: The root environment that mounts and manages other environments, providing a centralized entry point.
- **Specialized Environments**: Purpose-specific environments that encapsulate related tools and functionalities:
  - **Context Environment**: Manages conversation context and message history.
  - **Messaging Environment**: Handles communication with various messaging platforms.
  - **Web Environment**: Provides tools for web browsing, search, and HTTP requests.
  - **File Environment**: Manages file system operations like reading, writing, listing, and deleting files.

### Agent System

The agent system leverages the environment-based architecture:

- **Agent**: Processes user messages, uses the interface layer to execute tools, and generates responses.
- **Message Handler**: Routes messages to the appropriate agent and manages the processing flow.
- **Adapters**: Normalize messages from different platforms into a standard format.

## Key Features

- **Hierarchical Environment Structure**: Environments can be mounted within other environments, creating a flexible tool inheritance system.
- **Dynamic Tool Availability**: Tools become available or unavailable based on the mounting status of their parent environments.
- **Unified Interface**: The interface layer provides a consistent API for tool execution across all environments.
- **Modular Design**: Easy to extend with new environments and tools without modifying existing code.
- **Context Management**: Integrated context handling through the environment system.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Demo

```bash
python -m bot_framework.examples.context_demo
```

### Running Tests

```bash
python -m unittest bot_framework.tests.test_integration
```

## Example Usage

```python
from bot_framework.environments.manager import EnvironmentManager
from bot_framework.environments.system import SystemEnvironment
from bot_framework.environments.messaging import MessagingEnvironment
from bot_framework.interface.layer import InterfaceLayer

# Initialize environment system
environment_manager = EnvironmentManager()
system_env = SystemEnvironment()
messaging_env = MessagingEnvironment()

# Register environments
environment_manager.register_environment(system_env)
environment_manager.register_environment(messaging_env)

# Mount environments
system_env.mount_environment("messaging")

# Create interface layer (which includes agent functionality)
interface_layer = InterfaceLayer(environment_manager)

# Now you can use the interface layer to process messages
response = interface_layer.process_message({
    'chat_id': 'example_chat',
    'user_id': 'user123',
    'content': 'Hello, can you help me?',
    'role': 'user'
})

print(response['content'])
```

## Extending the Framework

### Creating a New Environment

```python
from bot_framework.environments.base import Environment
from typing import Dict, Any

class MyCustomEnvironment(Environment):
    def __init__(self):
        super().__init__(
            environment_id="custom",
            name="Custom Environment",
            description="A custom environment with specialized tools"
        )
        self._register_custom_tools()
    
    def _register_custom_tools(self):
        self.register_tool(
            "my_custom_tool",
            self.my_custom_tool,
            "Performs a custom operation",
            {
                "param1": "Description of parameter 1",
                "param2": "Description of parameter 2"
            }
        )
    
    def my_custom_tool(self, param1: str, param2: int) -> Dict[str, Any]:
        # Tool implementation
        return {"result": f"Processed {param1} {param2} times"}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 