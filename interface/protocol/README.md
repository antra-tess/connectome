# Custom Protocols

This directory contains the protocol classes used by the Bot Framework. Protocols define how the agent interacts with tools and formats both prompts and responses.

## Built-in Protocols

The framework includes several built-in protocols:

- **ReactProtocol**: Implements the ReAct (Reasoning, Action, Observation) pattern
- **FunctionCallingProtocol**: Uses native function calling capabilities in models that support it

## Creating Custom Protocols

You can create your own protocols without modifying the core framework code. The system will automatically discover and use your custom protocols based on configuration.

### Steps to Create a Custom Protocol

1. **Copy the template**:
   Copy `custom_protocol_template.py` to a new file with a descriptive name (e.g., `my_protocol.py`)

2. **Implement your protocol**:
   - Rename the class to match your protocol's purpose (e.g., `MyProtocol`)
   - Implement the required methods
   - Define the prompt format in the `protocol_prompt_format` property

3. **Update configuration**:
   Add your protocol to the configuration in `config.py`:

   ```python
   DEFAULT_PROTOCOLS = [
       # ... existing protocols ...
       {
           "name": "my_protocol",   # Used to reference the protocol
           "class": "MyProtocol",   # The exact class name from your file
           "enabled": True
       }
   ]
   ```

4. **Set as default (optional)**:
   To use your protocol as the default, update the `DEFAULT_PROTOCOL` setting:

   ```python
   DEFAULT_PROTOCOL = "my_protocol"
   ```

5. **Restart the framework**:
   Your protocol will be automatically discovered and used if selected.

### Example Template

The `custom_protocol_template.py` file provides a starting point with the basic structure and methods you need to implement. Key methods include:

- `format_system_prompt`: Format the system prompt with tool descriptions
- `extract_tool_calls`: Extract tool calls from LLM responses
- `format_tool_result`: Format tool results for the LLM
- `extract_final_response`: Extract the final response from LLM output
- `format_for_litellm`: Format messages for LiteLLM API calls

### Listing Available Protocols

You can use the provided utility script to list all available protocol classes:

```bash
python -m bot_framework.utils.list_protocols
```

This will show all protocol classes that the framework can discover.

## Protocol Selection

The framework uses a single protocol at a time, specified in the configuration:

- The `DEFAULT_PROTOCOL` setting determines which protocol is used by default
- Only protocols marked as `enabled: true` in the configuration are available
- If the default protocol is not enabled, the first enabled protocol is used

## Configuration Options

Each protocol entry in the configuration has these properties:

- `name`: Identifier used to reference the protocol
- `class`: The class name (must match exactly)
- `enabled`: Whether this protocol should be available (default: True)

You can customize these settings in `config.py` or via environment variables.

## Advanced: Environment Variables

You can also configure protocols via environment variables:

```
PROTOCOLS='[{"name":"custom","class":"CustomProtocol","enabled":true}]'
```

This format allows for runtime configuration without modifying the codebase. 