# Werewolf Game Configuration

# Available models and their configurations
AVAILABLE_MODELS = {
    # --- OpenAI Models ---
    "gpt-4o": {
        "name": "gpt-4o",
        "description": "OpenAI GPT-4o - latest flagship reasoning model",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "openai",
        "backend": "openai_compat"
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "description": "OpenAI GPT-4o Mini - faster, cheaper, lower latency",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "openai",
        "backend": "openai_compat"
    },

    # --- Google Models ---
    "gemini-pro": {
        "name": "gemini-pro", 
        "description": "Gemini Pro - balanced performance",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "google",
        "backend": "openai_compat"
    },
    "gemini-1.5-pro": {
        "name": "gemini-1.5-pro",
        "description": "Gemini 1.5 Pro - enhanced reasoning",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "google",
        "backend": "openai_compat"
    },
    "gemini-1.5-flash": {
        "name": "gemini-1.5-flash",
        "description": "Gemini 1.5 Flash - fast and efficient",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "google",
        "backend": "openai_compat"
    },
    
    # --- Local LLaMA Models ---
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "LLaMA 3.1 8B Instruct - local execution with activation capture",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "local",
        "backend": "llama_hf",
        "device": "cuda",
        "dtype": "float16",
        "load_in_8bit": False,
        "load_in_4bit": False,
    },
    "llama-3.1-8b-4bit": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "LLaMA 3.1 8B Instruct - 4-bit quantized for lower memory",
        "temperature": 0.7,
        "max_tokens": None,
        "provider": "local",
        "backend": "llama_hf",
        "device": "auto",
        "dtype": "float16",
        "load_in_4bit": True,
    },
}

# Default model
DEFAULT_MODEL = "llama-3.1-8b"

# Game settings
GAME_CONFIG = {
    "max_debate_turns": 6,
    "player_names": ["Alice", "Bob", "Charlie"],
    "default_roles": {
        "Alice": "Doctor",
        "Bob": "Werewolf",
        "Charlie": "Seer"
    }
}

# Environment settings
ENV_CONFIG = {
    "google_api_key_env": "GOOGLE_API_KEY",  # For Gemini models
    "openai_api_key_env": "OPENAI_API_KEY",  # For OpenAI models
    "debug_mode": False,
    "log_level": "INFO"
}

# Activation logging configuration
ACTIVATION_CONFIG = {
    "enabled": True,  # Enable activation logging (only works with local models)
    "layers": "all",  # 'all' or list of layer indices e.g. [0, 8, 16, 24, 31]
    "reduction": "last_token",  # 'last_token', 'mean_pool', 'output_tokens', or 'none'
    "save_format": "npz",  # File format for activation storage
    "compress": True,  # Use compression when saving
}
