import logging
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

logger = logging.getLogger(__name__)

# Environment variable prefix (optional but good practice)
ENV_PREFIX = "CONNECTOME_" 

class LLMConfig(BaseSettings):
    """Configuration for the LLM Provider."""
    # Define fields corresponding to your llm_provider config
    type: str = Field(default="litellm", description="Type of LLM provider (e.g., \'litellm\')")
    default_model: str = Field(default="gpt-4", description="Default model name")
    api_key: Optional[str] = Field(default=None, description="API Key for the LLM service (e.g., OpenAI)")
    # Add other provider-specific fields as needed, e.g.:
    # api_base: Optional[str] = Field(default=None, description="API Base URL (if needed)")
    
    # Use SettingsConfigDict for nested loading and prefixes
    model_config = SettingsConfigDict(env_prefix=f'{ENV_PREFIX}LLM_')

class ActivityAdapterConfig(BaseSettings):
    """Configuration for a single Activity Adapter connection."""
    id: str = Field(description="Unique identifier for this adapter connection")
    url: str = Field(description="URL of the adapter server (e.g., \'http://localhost:5001\')")
    # Auth could be a simple token or a nested model if more complex
    auth_token: Optional[str] = Field(default=None, description="Authentication token (if required)") 

    # Note: For lists like this, Pydantic usually expects JSON strings in the env var.
    # We'll handle parsing this in the main HostSettings.
    model_config = SettingsConfigDict(env_prefix=f'{ENV_PREFIX}ADAPTER_') # Prefix for individual vars if needed

class AgentConfig(BaseSettings):
    """Configuration for a single Agent."""
    agent_id: str = Field(description="Unique identifier for the agent")
    name: str = Field(description="Human-readable name for the agent")
    description: str = Field(default="An agent instance.", description="Description of the agent\'s purpose")
    agent_loop_component_type_name: str = Field(default="SimpleRequestResponseLoopComponent", description="Component type name for the agent\'s loop")
    platform_aliases: Dict[str, str] = Field(default_factory=dict, 
                                            description="Platform-specific user-facing aliases for the agent, "
                                                        "e.g., {\'discord_adapter_1\': \'MyBotName\'} for mention detection.")
    handles_direct_messages_from_adapter_ids: List[str] = Field(default_factory=list,
                                                               description="List of source_adapter_ids for which this agent "
                                                                           "will handle incoming direct messages.")
    # llm_provider_config_override: Optional[LLMConfig] = None # Example of nested override
    # inner_space_extra_components: List[str] = [] # Component type names
    
    # Note: Handling lists of agents via env vars is complex. Usually loaded from JSON/YAML.
    # We'll keep the structure but note the loading challenge.
    model_config = SettingsConfigDict(env_prefix=f'{ENV_PREFIX}AGENT_') # Prefix for individual vars if needed

class HostSettings(BaseSettings):
    """Main configuration settings loaded from environment variables."""
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    log_format: str = Field(default='%(asctime)s - %(name)s - %(levelname)s - %(message)s', description="Logging format string")
    log_to_file: bool = Field(default=False, description="Enable logging to file with rotation")
    log_file_path: str = Field(default="logs/connectome.log", description="Path to the log file (directory will be created)")
    log_max_lines_per_file: int = Field(default=5000, description="Maximum lines per log file before rotation")
    log_max_files: int = Field(default=10, description="Maximum number of log files to keep")
    
    # LLM Settings (will load from CONNECTOME_LLM_TYPE, CONNECTOME_LLM_DEFAULT_MODEL etc.)
    llm_provider: LLMConfig = Field(default_factory=LLMConfig)

    llm_type: str = Field(default="litellm", description="Type of LLM provider (e.g., \'litellm\')")
    llm_default_model: str = Field(default="gpt-4", description="Default model name")
    llm_api_key: Optional[str] = Field(default=None, description="API Key for the LLM service (e.g., OpenAI)")

    # Activity Client Adapters - Expects a JSON string in CONNECTOME_ACTIVITY_ADAPTERS_JSON
    activity_client_adapter_configs_json: str = Field(default='[{"id": "discord_adapter_1", "url": "http://localhost:5001", "auth_token": null}]', alias="ACTIVITY_ADAPTERS_JSON", description="JSON string representing a list of ActivityAdapterConfig objects")
    
    # Agents - Expects a JSON string in CONNECTOME_AGENTS_JSON
    agents_json: str = Field(
        alias="AGENTS_JSON",
        description="JSON string representing a list of AgentConfig objects"
    )

    # Parsed lists (populated after initialization)
    activity_client_adapter_configs: List[ActivityAdapterConfig] = []
    agents: List[AgentConfig] = []

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file='.env',        # Load from .env file
        env_prefix=ENV_PREFIX,  # Prefix for environment variables
        extra='ignore'          # Ignore extra fields found in env
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Parse JSON strings after loading
        try:
            import json
            # Parse adapters
            adapters_data = json.loads(self.activity_client_adapter_configs_json)
            self.activity_client_adapter_configs = [ActivityAdapterConfig(**item) for item in adapters_data]
            
            # Parse agents
            agents_data = json.loads(self.agents_json)
            self.agents = [AgentConfig(**item) for item in agents_data]
        except ImportError:
             logger.error("Failed to import \'json\'. Cannot parse JSON config strings.")
        except Exception as e:
            logger.error(f"Failed to parse JSON configuration strings (ACTIVITY_ADAPTERS_JSON or AGENTS_JSON): {e}", exc_info=True)
            # Keep empty lists if parsing fails

# Helper function to load settings
def load_settings() -> HostSettings:
    logger.info(f"Loading host configuration from .env file and environment variables (prefix: \'{ENV_PREFIX}\')...")
    try:
        settings = HostSettings()
        logger.info("Host configuration loaded successfully.")
        # Log loaded adapter and agent counts
        logger.info(f"Found {len(settings.activity_client_adapter_configs)} activity adapter configs.")
        logger.info(f"Found {len(settings.agents)} agent configs.")
        return settings
    except Exception as e:
        logger.exception(f"Critical error loading host configuration: {e}")
        # Decide how to handle failure - exit? Return defaults?
        # For now, let\'s re-raise to halt startup if config is broken
        raise ValueError(f"Failed to load configuration: {e}") from e 