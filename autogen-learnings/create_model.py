from autogen_ext.models.openai import OpenAIChatCompletionClient

def get_model_client() -> OpenAIChatCompletionClient:  # type: ignore
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model="ollama/llama3-groq-tool-use:8b", # replace with your model 
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://0.0.0.0:4000", # replace for the hosted model URL.
        temperature=0.1,
        max_retries=3,
        seed=42,
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )


