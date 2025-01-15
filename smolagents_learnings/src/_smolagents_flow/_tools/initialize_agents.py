from smolagents import (
    CodeAgent,
    ManagedAgent,
    ToolCallingAgent,
    MultiStepAgent,
)

class Initialize_Agent:
    """Initialize a available agent with the provided tools, model, and system prompt."""

    def __init__(self, model, tools: list = [], system_prompt: str = None, max_steps: int = 3, verbose: bool = False):
        """
        Initialize an instance of Initialize_Agent with the provided model, tools, system prompt, maximum steps, and verbosity.

        Parameters:
        - model (str): The model to be used for the agents.
        - tools (list): A list of tools to be provided to the agents.
        - max_steps (int, optional): The maximum number of steps for the agents. Defaults to 2.
        - verbose (bool, optional): A flag indicating whether to enable verbose mode for the agents. Defaults to False.
        """
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose

    def create_tools_agent(self, system_prompt: str = None, planning_interval: int = None) -> ToolCallingAgent:
        """
        Initialize a ToolCallingAgent with the provided tools, model, system prompt, planning interval, and verbosity.

        Parameters:
        - system_prompt (str, optional): The system prompt to be used for the agent. Defaults to None.
        - planning_interval (int, optional): The interval at which the agent should plan and execute actions.
            If not provided, the agent will plan and execute actions at every step. Defaults to None.

        Returns:
        - ToolCallingAgent: An instance of ToolCallingAgent initialized with the provided parameters.

        Raises:
        - Exception: If an error occurs during the initialization of the ToolCallingAgent.
        """
        try:
            agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                system_prompt=system_prompt,
                planning_interval=planning_interval,
                max_steps = self.max_steps,
                verbose=self.verbose,
                add_base_tools=True, # Required if no tools as passed.
            )
            return agent
        except Exception as e:
            raise Exception(f"Failed to create Tool Calling Agent.\nError: {str(e)}")
        
    def create_managed_agent(self, agent: list, name: str, description: str, managed_agent_prompt: str = None) -> CodeAgent:
        """
        Initialize a ManagedAgent with the provided agent, name, description, and managed agent prompt.

        Parameters:
        - agent (list): The agent to be managed.
        - name (str): The name of the managed agent.
        - description (str): A description of the managed agent.
        - managed_agent_prompt (str, optional): The managed agent prompt. Defaults to None.

        Returns:
        - CodeAgent: An instance of CodeAgent initialized with the provided parameters.

        Raises:
        - Exception: If an error occurs during the initialization of the CodeAgent.
        """
        try:
            agent = ManagedAgent(
                agent = agent,
                name=name,
                description=description,
                provide_run_summary=False,
                managed_agent_prompt=managed_agent_prompt,
                additional_prompting=None,
            )

            return agent
        except Exception as e:
            raise Exception(f"Failed to create CodeAgent.\nError: {str(e)}")
        
    def create_code_agent(self, system_prompt: str = None, authorized_imports: list = None, planning_interval: int = None, managed_agents: list = None) -> CodeAgent:
        """
        Initialize a CodeAgent with the provided tools, model, system prompt, maximum steps, authorized imports, planning interval, verbosity, and managed agents.

        Parameters:
        - system_prompt (str, optional): The system prompt to be used for the agent. Defaults to None.
        - authorized_imports (list, optional): A list of authorized imports for the agent. Defaults to None.
        - planning_interval (int, optional): The interval at which the agent should plan and execute actions.
            If not provided, the agent will plan and execute actions at every step. Defaults to None.
        - managed_agents (list, optional): A list of managed agents to be managed by the CodeAgent. Defaults to None.

        Returns:
        - CodeAgent: An instance of CodeAgent initialized with the provided parameters.

        Raises:
        - Exception: If an error occurs during the initialization of the CodeAgent.
        """
        try:
            agent = CodeAgent(
                tools=self.tools,
                model=self.model,
                system_prompt=system_prompt,
                additional_authorized_imports=authorized_imports,
                planning_interval=planning_interval,
                managed_agents=managed_agents,
                max_steps = self.max_steps,
                verbose=self.verbose,
                # add_base_tools=True,
            )

            return agent
        except Exception as e:
            raise Exception(f"Failed to create CodeAgent.\nError: {str(e)}")