# Custom Environments

This directory contains the environment classes used by the Bot Framework. Environments are the contextual layer that creates coherent representations of activities and states for agents.

## Built-in Environments

The framework includes several built-in environments:

- **SystemEnvironment**: The root environment that contains all other environments
- **WebEnvironment**: For web interactions and searches
- **MessagingEnvironment**: For handling messages across platforms
- **FileEnvironment**: For file system interactions

## Creating Custom Environments

You can create your own environments without modifying the core framework code. The system will automatically discover and use your custom environments based on configuration.

### Steps to Create a Custom Environment

1. **Copy the template**:
   Copy `custom_environment_template.py` to a new file with a descriptive name (e.g., `my_specialized_environment.py`)

2. **Implement your environment**:
   - Rename the class to match your environment's purpose
   - Implement any required methods
   - Add custom tools using the `register_tool` method
   - Add custom state and rendering logic

3. **Update configuration**:
   Add your environment to the configuration in `config.py`:

   ```python
   DEFAULT_ENVIRONMENTS = [
       # ... existing environments ...
       {
           "id": "my_env",
           "class": "MySpecializedEnvironment",  # The exact class name from your file
           "enabled": True,
           "mount_point": None  # Optional custom mount point
       }
   ]
   ```

4. **Restart the framework**:
   Your environment will be automatically discovered and initialized.

### Example Template

The `custom_environment_template.py` file provides a starting point with the basic structure and methods you need to implement. Key methods include:

- `__init__`: Initialize your environment and its state
- `_register_tools`: Register tools provided by your environment
- `get_state`: Return the current state of your environment (machine-readable representation)
- `render`: Render a representation of your environment's structure and capabilities
- `render_state_for_context`: Format the current state for inclusion in the agent's context (human-readable representation)

### Understanding State vs. Context Rendering

The framework distinguishes between different ways of representing an environment:

1. **`get_state()`**: Returns a raw dictionary with the environment's current internal state. Used primarily for internal operations.

2. **`render()`**: Provides a complete structural representation of the environment, focusing on its capabilities and architecture. Used for describing what the environment can do.

3. **`render_state_for_context()`**: Creates a human-readable, formatted representation specifically designed for inclusion in the agent's context. This is what the LLM "sees" about your environment, so it should be informative and well-structured.

When creating a custom environment, implementing `render_state_for_context()` is particularly important as it directly affects how the agent understands and interacts with your environment.

### Listing Available Environments

You can use the provided utility script to list all available environment classes:

```bash
python -m bot_framework.utils.list_environments
```

This will show all environment classes that the framework can discover.

## Environment Hierarchy

The framework uses a hierarchical environment structure:

- The **SystemEnvironment** is always created and serves as the root
- Other environments are mounted to the system environment
- Each environment can have its own tools and capabilities
- The system environment provides tools for navigating between environments

## Configuration Options

Each environment entry in the configuration has these properties:

- `id`: Unique identifier for this environment instance
- `class`: The class name (must match exactly)
- `enabled`: Whether to initialize this environment (default: True)
- `mount_point`: Optional custom mount path in the system environment (default: same as id)

You can customize these settings in `config.py` or via environment variables.

## Advanced: Environment Variables

You can also configure environments via environment variables:

```
ENVIRONMENTS='[{"id":"custom_env","class":"MyCustomEnvironment","enabled":true,"mount_point":"custom"}]'
```

This format allows for runtime configuration without modifying the codebase. 