from game_graph import graph, GameState  
from player import Player             
from langchain_openai import ChatOpenAI
import os
import argparse
from dotenv import load_dotenv
from logs import init_logging_state, write_final_state, print_header, print_subheader, print_kv, write_final_metrics
from config import AVAILABLE_MODELS

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


class HuggingFaceLLM:
    """Wrapper for HuggingFace models to match LangChain interface."""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get number of available GPUs
        n_gpus = torch.cuda.device_count()
        print(f"Loading model: {model_name}")
        print(f"Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optimize for multi-GPU with tensor parallelism
        # For 8B model on 3x A100 80GB, spread across all GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically distribute across all GPUs
            attn_implementation="flash_attention_2",  # Use Flash Attention 2 for speed
            low_cpu_mem_usage=True,
        )
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile for faster inference...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
        )
        
        # Print device map for verification
        if hasattr(self.model, 'hf_device_map'):
            print(f"Model device map: {self.model.hf_device_map}")
        print(f"Model loaded successfully")
    
    def invoke(self, prompt: str, max_tokens: int = None, timeout: int = None):
        """Generate response matching LangChain interface."""
        max_new_tokens = max_tokens or self.max_tokens
        
        messages = [{"role": "user", "content": prompt}]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        generated_text = outputs[0]["generated_text"][-1]["content"]
        
        class Response:
            def __init__(self, text):
                self.content = text
        
        return Response(generated_text)


def get_llm(model_name="gpt-4o", api_key=None):
    """Initialize the language model with configurable parameters."""
    os.environ["MODEL_NAME"] = model_name
    
    model_config = AVAILABLE_MODELS.get(model_name, {})
    provider = model_config.get("provider", "openai")
    
    if provider == "huggingface":
        hf_model_name = model_config.get("name", model_name)
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 512)
        return HuggingFaceLLM(
            model_name=hf_model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set and no API key provided")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.7
        )


def run_werewolf_game(model_name="gpt-4o", api_key=None, log_dir: str = "./logs", enable_file_logging: bool = True):
    """Run a werewolf game with the specified model."""
    print_header("Starting Werewolf Game")
    print_kv("Model", model_name)
    
    # Initialize the language model
    llm = get_llm(model_name, api_key)
    
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
        name: Player(name=name, role=roles[name], llm=llm)
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
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Werewolf Game with AI players")
    parser.add_argument(
        "--model", 
        default="gpt-4o",
        help="Model to use (default: gpt-4o). Options: gpt-4o, gpt-4o-mini, llama-3.1-8b, llama-3.1-70b"
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
    
    args = parser.parse_args()
    
    try:
        # If no API key provided via args, rely on environment variables loaded from .env
        final_state = run_werewolf_game(args.model, args.api_key, log_dir=args.log_dir, enable_file_logging=(not args.no_file_logging))

        print_subheader("Game Results")
        print_kv("Final alive players", final_state.alive_players, indent=2)
        if hasattr(final_state, 'winner'):
            print_kv("Winner", final_state.winner, indent=2)
        
    except Exception as e:
        print_subheader("Error")
        print_kv("Message", f"{e}")
        print_subheader("Troubleshooting")
        print_kv("1", "Make sure your OPENAI API key is valid", indent=2)
        print_kv("2", "Install dependencies: pip install -r requirements.txt", indent=2)
        print_kv("3", "Try a different model: python run.py --model gpt-4o-mini", indent=2)