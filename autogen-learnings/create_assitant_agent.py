from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

import shutil
import subprocess
import json
import asyncio

from create_model import get_model_client

# Helper Fn: Check if a tool is installed
def is_tool_installed(tool_name: str) -> bool:
    """
    Check if a tool is installed on the system.
    """
    return shutil.which(tool_name) is not None

# Helper Fn: Ping Host
async def ping_host(host:str) -> str:
    """
    Pings a specified host using the system's ping utility. It verifies the availability of the ping tool before execution and handles various error cases including timeouts and network unreachability.

    """

    # First verify ping utility is available
    if is_tool_installed("ping"):
        try:
            # Execute ping with specified parameters
            output = subprocess.run(
                ['ping', '-c', "3", host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Process the ping results
            if output.returncode == 0:
                # Host is up and responding
                return json.dumps({
                    "host": host,
                    "status": "reachable", 
                    "details": output.stdout.strip()
                })
            else:
                # Host is down or unreachable
                return json.dumps({
                    "host": host,
                    "status": "unreachable",
                    "details": output.stdout.strip()
                })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "timeout",
                "details": "Ping request timed out."
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "details": str(e)
            })
    else:
        return json.dumps({
            "status": "tool not installed",
            "details": "Ping tool is not installed on the system."
        })



# Create an agent that uses the OpenAI GPT-4o model.
model_client = get_model_client()

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[ping_host],
    system_message="Use tools to solve tasks.",
)

async def handle_message(message: TextMessage, token: CancellationToken) -> TextMessage:
    """
    Handle incoming messages and provide responses using the defined tools.
    """
    if message.content.startswith("ping"):
        host = message.content.split(" ")[1]
        response = await agent.call_tool("ping_host", host)
        return TextMessage(response)
    else:
        return TextMessage("I'm sorry, I can't help you with that.")
    

# Run the agent in an infinite loop.
while True:
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        break

