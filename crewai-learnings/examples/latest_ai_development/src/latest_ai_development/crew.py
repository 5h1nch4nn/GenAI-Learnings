from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from dotenv import load_dotenv

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

load_dotenv()

@CrewBase
class LatestAiDevelopment():
	"""LatestAiDevelopment crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	llm = LLM(
		model = "ollama/llama3.2:3b",
		base_url="http://localhost:11434"
	)

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'], # Load the key paramters from the yaml file
			llm=self.llm,
			function_calling_llm=None,  # Optional: Separate LLM for tool calling
			memory=True,  # Default: True
			verbose=True,  # Default: False
			allow_delegation=False,  # Default: False
			max_iter=5,  # Default: 20 iterations
			max_rpm=None,  # Optional: Rate limit for API calls
			max_execution_time=300,  # Optional: Maximum execution time in seconds
			max_retry_limit=2,  # Default: 2 retries on error
			allow_code_execution=False,  # Default: False
			code_execution_mode="unsafe",  # Default: "safe" (options: "safe", "unsafe")
			respect_context_window=True,  # Default: True
			use_system_prompt=True,  # Default: True
			tools=[SerperDevTool()],  # Optional: List of tools
			knowledge_sources=None,  # Optional: List of knowledge sources
			embedder_config=None,  # Optional: Custom embedder configuration
			system_template=None,  # Optional: Custom system prompt template
			prompt_template=None,  # Optional: Custom prompt template
			response_template=None,  # Optional: Custom response template
			step_callback=None,  # Optional: Callback function for monitoring
		)

	@agent
	def coder(self) -> Agent:
		return Agent(
			config=self.agents_config['coder'],
			llm=self.llm,
			verbose=True,
		)
	

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			llm=self.llm,
			max_iter=5,  # Default: 20 iterations
			max_rpm=None,  # Optional: Rate limit for API calls
			max_execution_time=300,  # Optional: Maximum execution time in seconds
			verbose=True,
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
		)
	
	@task
	def coding_task(self) -> Task:
		return Task(
			config=self.tasks_config['coding_task'],
        )

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
