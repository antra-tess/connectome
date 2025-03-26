# Elements and Spaces

This directory contains the element and space classes used by the Bot Framework. Elements and Spaces create coherent representations of activities and states for agents, with Spaces managing the lifecycle and timeline context of Elements.

## Built-in Elements and Spaces

The framework includes several built-in elements and spaces:

- **SystemSpace**: The root space that contains all other elements
- **WebElement**: For web interactions and searches
- **MessagingElement**: For handling messages across platforms
- **FileElement**: For file system interactions

## Element Base Class

The `BaseElement` class provides a foundation for all elements with common functionality:

- Element identification (ID, name, description)
- Tool registration and execution
- Observer pattern for state updates
- Hierarchical element mounting
- State management
- Timeline context handling

Key methods include:

- `register_tool`: Register a function as a tool available to the agent
- `execute_tool`: Execute a registered tool
- `get_state`: Get the current state of the element
- `update_state`: Update the state based on incoming data
- `register_observer` / `register_edit_observer`: Register observers for state updates
- `notify_observers`: Notify all registered observers of a state update

## Element State Management

Elements are responsible for maintaining their internal state but not for formatting it. The key method is:

**`get_state()`**: Returns a dictionary representing the element's internal state. This is the raw data structure that will be used by the interface layer.

The formatting of element states for agent consumption is handled entirely by the interface layer.

## Types of Observers

Elements support two types of observers:

1. **Full Observers**: Receive all updates when the element is mounted.
2. **Edit Observers**: Receive only edit-related updates, even after the element is unmounted.

This dual observer system supports the interface layer's dual storage requirements, where:
- All element data is stored while the element is mounted
- Only edit-related updates are stored after unmounting

## Creating Custom Elements

To create a custom element:

1. Subclass the `BaseElement` base class
2. Define your element's specific `EVENT_TYPES` and `ATTENTION_SIGNALS`
3. Implement `get_state()` to return your element's state
4. Implement `update_state()` to handle incoming events
5. Register any tools your element provides

See the `custom_element_template.py` file for a complete example.

Here's a simple skeleton:

```python
from elements.base import BaseElement

class CustomElement(BaseElement):
    # Define event types this element handles
    EVENT_TYPES = ['custom_event_type']
    
    # Define signals that require agent attention
    ATTENTION_SIGNALS = ['important_signal']
    
    def get_state(self) -> Dict[str, Any]:
        """Return the current state of this element."""
        return {
            'id': self.id,
            'name': self.name,
            # Add element-specific state
        }
    
    def update_state(self, update_data: Dict[str, Any]) -> bool:
        """Update element state based on incoming data."""
        # Handle the update
        return True
```

### Steps to Create a Custom Element

1. **Copy the template**:
   Copy `custom_element_template.py` to a new file with a descriptive name (e.g., `my_specialized_element.py`)

2. **Implement your element**:
   - Rename the class to match your element's purpose
   - Implement any required methods
   - Add custom tools using the `register_tool` method
   - Add custom state and rendering logic

3. **Update configuration**:
   Add your element to the configuration in `config.py`:

   ```python
   DEFAULT_ELEMENTS = [
       # ... existing elements ...
       {
           "id": "my_element",
           "class": "MySpecializedElement",  # The exact class name from your file
           "enabled": True,
           "mount_point": None  # Optional custom mount point
       }
   ]
   ```

4. **Restart the framework**:
   Your element will be automatically discovered and initialized.

### Example Template

The `custom_element_template.py` file provides a starting point with the basic structure and methods you need to implement. Key methods include:

- `__init__`: Initialize your element and its state
- `_register_tools`: Register tools provided by your element
- `get_state`: Return the current state of your element (machine-readable representation)
- `render`: Render a representation of your element's structure and capabilities
- `render_state_for_context`: Format the current state for inclusion in the agent's context (human-readable representation)

### Understanding State vs. Context Rendering

The framework distinguishes between different ways of representing an element:

1. **`get_state()`**: Returns a raw dictionary with the element's current internal state. Used primarily for internal operations.

2. **`render()`**: Provides a complete structural representation of the element, focusing on its capabilities and architecture. Used for describing what the element can do.

3. **`render_state_for_context()`**: Creates a human-readable, formatted representation specifically designed for inclusion in the agent's context. This is what the LLM "sees" about your element, so it should be informative and well-structured.

When creating a custom element, implementing `render_state_for_context()` is particularly important as it directly affects how the agent understands and interacts with your element.

### Listing Available Elements

You can use the provided utility script to list all available element classes:

```bash
python -mconnectome.utils.list_elements
```

This will show all element classes that the framework can discover.

## Element Hierarchy

The framework uses a hierarchical element structure:

- The **SystemSpace** is always created and serves as the root
- Other elements are mounted to the system space
- Each element can have its own tools and capabilities
- The system space provides tools for navigating between elements

## Configuration Options

Each element entry in the configuration has these properties:

- `id`: Unique identifier for this element instance
- `class`: The class name (must match exactly)
- `enabled`: Whether to initialize this element (default: True)
- `mount_point`: Optional custom mount path in the system space (default: same as id)

You can customize these settings in `config.py` or via environment variables.

## Advanced: Element Variables

You can also configure elements via environment variables:

```
ELEMENTS='[{"id":"custom_element","class":"MyCustomElement","enabled":true,"mount_point":"custom"}]'
```

This format allows for runtime configuration without modifying the codebase. 