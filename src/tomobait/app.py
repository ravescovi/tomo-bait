from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agents import (
    LLMNotConfiguredError,
    get_llm_status,
    reset_agents,
    run_agent_chat,
)
from .config import (
    TomoBaitConfig,
    backup_config,
    get_config,
    reload_config,
    save_config,
)
from .config_generator import config_dict_to_yaml, generate_config_from_prompt

load_dotenv()

# Load configuration
config = get_config()

# --- FastAPI App ---
api = FastAPI()


class ChatQuery(BaseModel):
    query: str


class ConfigResponse(BaseModel):
    config: dict


class GenerateConfigRequest(BaseModel):
    prompt: str


class GenerateConfigResponse(BaseModel):
    yaml_config: str
    config_dict: dict


@api.get("/health")
async def health_check():
    """
    Health check endpoint that reports LLM availability.
    Returns 200 even if LLM is not configured, but indicates status.
    """
    llm_status = get_llm_status()
    return {
        "status": "healthy",
        "llm": llm_status,
    }


@api.post("/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to receive a query and return the agent's response.
    """
    try:
        answer = run_agent_chat(chat_query.query)
        return {"response": answer}
    except LLMNotConfiguredError as e:
        # Specific error for missing API key - return 503 (Service Unavailable)
        llm_status = get_llm_status()
        api_key_env = llm_status['api_key_env']
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_not_configured",
                "message": f"LLM not available: {str(e)}",
                "hint": f"Set {api_key_env} or configure a different provider",
                "provider": llm_status["provider"],
                "model": llm_status["model"],
            }
        )
    except Exception as e:
        error_msg = str(e)
        # Check if it's an API overload error
        if "503" in error_msg or "overloaded" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail="The AI service is currently overloaded. "
                "Please try again in a moment."
            )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment before trying again."
            )
        else:
            # Generic error
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing your request: {error_msg}"
            )


@api.get("/config")
async def get_config_endpoint():
    """
    Get current configuration.
    """
    return {"config": config.model_dump()}


@api.post("/config")
async def update_config_endpoint(new_config: dict):
    """
    Update configuration (requires restart to apply).
    """
    # This is a placeholder - in production you'd want to validate and save
    return {"message": "Configuration updated. Restart backend to apply changes."}


@api.post("/generate-config")
async def generate_config_endpoint(request: GenerateConfigRequest):
    """
    Generate a configuration from natural language prompt using Gemini.
    """
    try:
        # Generate config using Gemini
        config_dict = generate_config_from_prompt(request.prompt)

        # Convert to YAML
        yaml_config = config_dict_to_yaml(config_dict)

        return GenerateConfigResponse(
            yaml_config=yaml_config,
            config_dict=config_dict
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate config: {str(e)}"
        )


@api.post("/apply-config")
async def apply_config_endpoint(new_config: dict):
    """
    Apply a new configuration after backing up the old one.
    """
    try:
        # Backup current config
        backup_path = backup_config()

        # Validate and save new config
        validated_config = TomoBaitConfig(**new_config)
        save_config(validated_config)

        # Reset agents so they pick up new config on next use
        reset_agents()

        return {
            "message": "Configuration applied successfully!",
            "backup_path": backup_path
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


@api.post("/reset-agents")
async def reset_agents_endpoint():
    """
    Reset agents to pick up new configuration.
    Useful after changing LLM settings.
    """
    reset_agents()
    llm_status = get_llm_status()
    return {
        "message": "Agents reset. Will reinitialize on next chat request.",
        "llm": llm_status,
    }


# --- Startup Event ---
@api.on_event("startup")
async def startup_event():
    """Log startup information."""
    # Log LLM status on startup
    llm_status = get_llm_status()
    if llm_status["available"]:
        print(f"✅ LLM configured: {llm_status['provider']}/{llm_status['model']}")
    else:
        print(f"⚠️  LLM not available: {llm_status['error']}")
        print(f"   Set {llm_status['api_key_env']} or configure a different provider")
