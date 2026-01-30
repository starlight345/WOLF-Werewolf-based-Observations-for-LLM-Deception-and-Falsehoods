from game_graph import graph, GameState  
from player import Player             
from langchain_openai import ChatOpenAI
import os
import argparse
from dotenv import load_dotenv
from logs import init_logging_state, write_final_state, print_header, print_subheader, print_kv, write_final_metrics
from config import AVAILABLE_MODELS, DEFAULT_MODEL, ACTIVATION_CONFIG
from llm_backends import LlamaHFBackend, OpenAICompatBackend, BaseLLMBackend

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def get_llm(model_name=None, api_key=None):
    """
    Initialize the language model with configurable parameters.
    
    Supports both API-based models (OpenAI, Gemini) and local models (LLaMA).
    Returns the appropriate backend based on model configuration.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    # Get model config
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_config = AVAILABLE_MODELS[model_name].copy()
    backend_type = model_config.get("backend", "openai_compat")
    
    os.environ["MODEL_NAME"] = model_name
    
    if backend_type == "llama_hf":
        # Local LLaMA backend with activation capture
        print(f"Initializing LLaMA backend: {model_config['name']}")
        
        backend_config = {
            "model_name_or_path": model_config["name"],
            "device": model_config.get("device", "cuda"),
            "dtype": model_config.get("dtype", "float16"),
            "load_in_8bit": model_config.get("load_in_8bit", False),
            "load_in_4bit": model_config.get("load_in_4bit", False),
            "activation_layers": ACTIVATION_CONFIG.get("layers", "all"),
            "activation_reduce": ACTIVATION_CONFIG.get("reduction", "last_token"),
            "max_length": 2048,
        }
        
        return LlamaHFBackend(backend_config)
    
    elif backend_type == "openai_compat":
        # API-based backend (OpenAI, Gemini via LangChain)
        print(f"Initializing OpenAI-compatible backend: {model_name}")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set, model initialization may fail")
        
        backend_config = {
            "model_name": model_config["name"],
            "temperature": model_config.get("temperature", 0.7),
            "api_key": api_key,
        }
        
        return OpenAICompatBackend(backend_config)
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def run_werewolf_game(model_name=None, api_key=None, log_dir: str = "./logs", enable_file_logging: bool = True, log_activations: bool = None):
    """
    Run a werewolf game with the specified model.
    
    Args:
        model_name: Name of the model to use (from AVAILABLE_MODELS)
        api_key: API key for API-based models
        log_dir: Directory to save logs
        enable_file_logging: Whether to enable file logging
        log_activations: Whether to log activations (default: from ACTIVATION_CONFIG)
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if log_activations is None:
        log_activations = ACTIVATION_CONFIG.get("enabled", True)
    
    print_header("Starting Werewolf Game")
    print_kv("Model", model_name)
    print_kv("Activation Logging", "Enabled" if log_activations else "Disabled")
    
    # Initialize the language model
    llm = get_llm(model_name, api_key)
    
    # Check if backend supports activation logging
    supports_activations = isinstance(llm, BaseLLMBackend) and hasattr(llm, 'config')
    if log_activations and not supports_activations:
        print("Warning: Activation logging requested but model backend doesn't support it")
        log_activations = False
    
    # Game setup
    players = ["Alice", "Bob", "Selena", "Raj", "Frank", "Joy", "Cyrus", "Emma"]
    roles = {
        "Alice": "Doctor",
        "Bob": "Werewolf", 
        "Selena": "Seer", 
        "Raj": "Villager", 
        "Frank": "Villager", 
        "Joy": "Werewolf", 
        "Cyrus": "Villager", 
        "Emma": "Villager"
    }

    seer = next((p for p in players if roles[p] == "Seer"), None)
    doctor = next((p for p in players if roles[p] == "Doctor"), None)
    werewolves = [p for p in players if roles[p] == "Werewolf"]
    villagers = [p for p in players if roles[p] == "Villager"]

    player_objects = {
        name: Player(name=name, role=roles[name], llm=llm, log_activations=log_activations)
        for name in players
    }

    initial_state = GameState(
        round_num=0,
        players=players,
        alive_players=players.copy(),
        roles=roles,
        villagers=villagers,
        werewolves=werewolves,
        seer=seer,
        doctor=doctor,
        phase="eliminate",
        game_logs=[],
        deception_history={},
        deception_scores={}
    )

    # Initialize file logging on the state
    initial_state = init_logging_state(initial_state, log_dir=log_dir, enable_file_logging=enable_file_logging)

    # Run the game
    print_subheader("Execute")
    print_kv("Action", "Compiling and running the game graph...")
    runnable = graph.compile()
    final_state = runnable.invoke(initial_state, config={
        "recursion_limit": 1000,
        "configurable": {
            "player_objects": player_objects,
            "MAX_DEBATE_TURNS": 6
        }
    })
    
    # Persist the final state to disk if logging is enabled
    write_final_state(final_state)
    # Persist organized final metrics (no raw prompts/outputs)
    write_final_metrics(final_state)

    print_subheader("Status")
    print_kv("Result", "Game completed successfully!")

    # Print helpful info for locating logs
    paths = getattr(final_state, "log_paths", {})
    if paths:
        print_subheader("Log Files")
        print_kv("Events (NDJSON)", paths.get('events'), indent=2)
        print_kv("Final State JSON", paths.get('state'), indent=2)
        print_kv("Final Metrics JSON", paths.get('metrics'), indent=2)
        print_kv("Run Metadata", paths.get('meta'), indent=2)
        if log_activations and paths.get('activations'):
            print_kv("Activations Directory", paths.get('activations'), indent=2)
    
    # Cleanup if using local model
    if isinstance(llm, BaseLLMBackend):
        llm.cleanup()
    
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Werewolf Game with AI players")
    parser.add_argument(
        "--model", 
        default=None,
        help=f"Model to use (default: {DEFAULT_MODEL}). Available: {', '.join(AVAILABLE_MODELS.keys())}"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (alternatively set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="Directory to store run logs (events NDJSON + final JSON). Default: ./logs"
    )
    parser.add_argument(
        "--no-file-logging",
        action="store_true",
        help="Disable writing logs to disk (events and final state)"
    )
    parser.add_argument(
        "--log-activations",
        action="store_true",
        default=None,
        help="Enable activation logging (only works with local models like LLaMA)"
    )
    parser.add_argument(
        "--no-log-activations",
        action="store_true",
        help="Disable activation logging"
    )
    
    args = parser.parse_args()
    
    # Determine activation logging setting
    log_activations = None
    if args.log_activations:
        log_activations = True
    elif args.no_log_activations:
        log_activations = False
    
    try:
        # If no API key provided via args, rely on environment variables loaded from .env
        final_state = run_werewolf_game(
            model_name=args.model,
            api_key=args.api_key,
            log_dir=args.log_dir,
            enable_file_logging=(not args.no_file_logging),
            log_activations=log_activations
        )

        print_subheader("Game Results")
        print_kv("Final alive players", final_state.alive_players, indent=2)
        if hasattr(final_state, 'winner'):
            print_kv("Winner", final_state.winner, indent=2)
        
    except Exception as e:
        print_subheader("Error")
        print_kv("Message", f"{e}")
        print_subheader("Troubleshooting")
        print_kv("1", "Install dependencies: pip install -r requirements.txt", indent=2)
        print_kv("2", "For API models: Set OPENAI_API_KEY environment variable", indent=2)
        print_kv("3", "For local models: Ensure you have GPU and CUDA installed", indent=2)
        print_kv("4", "List available models: check config.py AVAILABLE_MODELS", indent=2)
        import traceback
        traceback.print_exc()