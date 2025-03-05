# Bot Framework

A modular, extensible framework for building conversational AI agents with a three-layer architecture supporting dynamic environments, privileged and environment-specific tools, and context management.

## Architecture Overview

Bot Framework implements a three-layer architecture that separates concerns between external connections, environment representation, and agent cognition:

### 1. Activity Layer
The foundation that connects to external systems like messaging platforms and APIs:

- **SocketIO Adapters**: Normalize different messaging platforms (Slack, Discord, etc.)
- **Message Handling**: Routes normalized messages to appropriate environments
- **Activity Client**: Maintains connections to external services

### 2. Environment Layer
The contextual layer that creates coherent representations of activities and states:

- **Environment Manager**: Orchestrates registered environments and their hierarchical relationships
- **Environment Types**:
  - **System Environment**: Root environment that provides access to all mounted environments
  - **Messaging Environment**: Handles chat conversations across platforms
    - Publishes messages and typing indicators to registered observers
    - Maintains conversation state across different platforms
  - **File Environment**: File system operations and management
  - **Web Environment**: Internet browsing, API access, and search capabilities
  - **Custom Environments**: Extensible to add domain-specific capabilities

- **Environment Hierarchy**:
  - Environments can be mounted inside other environments
  - Tools become available through the mounting hierarchy
  - Each environment maintains its own state and context

### 3. Interface Layer
The cognitive layer that manages how the agent perceives and interacts with environments:

- **LLM Processor**: Handles interactions with language models
- **Message Processor**: Processes messages and executes tools
- **Context Handler**: Manages conversation history and environment state
- **Tool Manager**: Registers and manages privileged tools
- **Environment Renderer**: Formats environment states for agent consumption

## Key Features

### Environment-Based Architecture
- **Hierarchical Structure**: Environments can mount other environments, creating a tree structure
- **Dynamic Tool Availability**: Tools become available based on mounted environments
- **Environment State**: Each environment maintains its own state and context
- **Mount Points**: Environments attach to one another at specific points in the hierarchy

### Two-Tier Tool System
- **Privileged Tools** (Interface Layer):
  - Manage context and memory
  - Control environment rendering
  - Handle system-level operations
  - Independent of specific environments

- **Environment Tools** (Environment Layer):
  - Specific to particular environments
  - Provide domain-specific capabilities
  - Subject to environment permissions
  - Accessible through environment mounting

### Context Management
- **Conversation History**: Maintains chronological record of interactions
- **Environment States**: Captures current state of all environments
- **Context Summarization**: Compresses older context to maintain relevance
- **Multi-Environment Context**: Aggregates context from mounted environments

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Framework

```bash
python -m bot_framework.main
```

### Configuration

The framework uses a central `config.py` file for core settings:

```python
# bot_framework/config.py
LLM_API_KEY = "your-api-key"
LLM_MODEL = "your-model"
DEFAULT_PROTOCOL = "openai"  # or "anthropic", "llama" etc.
```

## Example Usage

### Basic Setup

```python
from bot_framework.environments.manager import EnvironmentManager
from bot_framework.environments.system import SystemEnvironment
from bot_framework.interface.layer.interface_layer import InterfaceLayer

# Initialize environment system
env_manager = EnvironmentManager()
system_env = SystemEnvironment()

# Register the root environment
env_manager.register_environment(system_env)

# Create interface layer
interface = InterfaceLayer(env_manager)

# Process a message
response = interface.process_message(
    user_id="user123",
    message_text="Hello, can you help me?",
    env_id="system"  # Specify the environment ID
)
```

### Creating a Custom Environment

```python
from bot_framework.environments.base import Environment
from typing import Dict, Any

class MyCustomEnvironment(Environment):
    def __init__(self, env_id: str = "custom_env", name: str = "Custom Environment"):
        super().__init__(env_id=env_id, name=name)
        self._register_tools()
    
    def _register_tools(self):
        @self.register_tool(
            name="custom_tool",
            description="Performs a specialized operation",
            parameter_descriptions={
                "param1": "First parameter",
                "param2": "Second parameter"
            }
        )
        def custom_tool(param1: str, param2: int) -> Dict[str, Any]:
            # Tool implementation
            return {"result": f"Processed {param1} {param2} times"}
```

### Mounting Environments

```python
# Create environments
messaging_env = MessagingEnvironment(env_id="messaging")
file_env = FileEnvironment(env_id="file_system")

# Register with manager
env_manager.register_environment(messaging_env)
env_manager.register_environment(file_env)

# Mount to system environment
system_env.mount(messaging_env, mount_point="messaging")
system_env.mount(file_env, mount_point="files")
```

## Architecture Principles

### Environment Hierarchy
- System environments contain mounted representations of other environments
- An agent can participate in multiple environments simultaneously
- The system environment mediates all interactions with mounted environments

### Separation of Concerns
- Activity layer handles external world interfaces
- Environment layer manages state representation and hierarchy
- Interface layer manages agent cognition and experience

### Message Routing
- Messages are associated with specific environments using `env_id`
- Tools execute within the context of specific environments
- Updates propagate through the environment hierarchy

### Observer Pattern for Layer Communication
- Environments publish events rather than directly calling external services
- Activity layer components register as observers for environment events
- Maintains clean separation between architectural layers
- Allows for flexible replacement of component implementations

## Extending the Framework

The framework is designed for extension at multiple levels:

1. **Custom Environments**: Create specialized environments for domain-specific functionality
2. **Custom Tools**: Register tools within environments or as privileged tools
3. **Custom Protocols**: Implement support for different LLM APIs
4. **Custom Adapters**: Connect to additional messaging platforms or external services

## Interaction diagram
sequenceDiagram
    participant User
    participant Adapter as Normalizing Adapter
    participant Activity as Activity Layer
    participant Environment as Environment Layer
    participant Interface as Interface Layer
    participant LLM as Language Model

    User->>Adapter: Send message
    Adapter->>Activity: Emit "chat_message" event
    Note over Adapter,Activity: Standardized message format

    Activity->>Environment: handle_message()
    Environment->>Environment: Update environment state
    Environment->>Interface: observe_message()
    
    Interface->>Interface: save_message()
    Interface->>Environment: build_agent_context()
    Environment->>Interface: Return environment context
    
    Interface->>Interface: Get available tools
    Interface->>LLM: process_with_context()
    LLM->>Interface: Return response
    
    Interface->>Interface: Parse tool calls (if any)
    
    alt Tool Execution
        Interface->>Environment: execute_tool()
        Environment->>Interface: Return tool result
        Interface->>LLM: process_with_tool_result()
        LLM->>Interface: Return updated response
    end
    
    Interface->>Environment: save_message() (assistant)
    Environment->>Activity: publish_message()
    
    Activity->>Adapter: Emit "bot_response" event
    Adapter->>User: Display response
    
    opt Typing Indicator
        Interface->>Environment: publish_typing_indicator()
        Environment->>Activity: send_typing_indicator()
        Activity->>Adapter: Emit "typing_indicator" event
        Adapter->>User: Show typing indicator
    end
