from smolagents import (
    HfApiModel,
    TransformersModel,
    LiteLLMModel,
)

from huggingface_hub import InferenceClient, login

from dotenv import load_dotenv
import os

load_dotenv()

class Initialize_Client:
    def __init__(self, model_id="Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        Initialize the Initialize_Model class with a default model_id.

        Parameters:
        model_id (str): The identifier of the model to be initialized.
            Default is "Qwen/Qwen2.5-Coder-32B-Instruct".

        Returns:
        None
        """
        self.model_id = model_id

    def initialize_Inference_client(self):
        """
        Initialize a client using the Inference Client model.

        This function initializes a client using the InferenceClient class from the Hugging Face Hub library.
        It takes no parameters and uses the model_id provided during class initialization.

        Parameters:
        None

        Returns:
        client (InferenceClient): An instance of the InferenceClient class representing the initialized client.
            If the initialization fails, an exception is raised.

        Raises:
        Exception: If the client initialization fails.
        """
        try:
            # Initialize the client here
            self.client = InferenceClient(
                model=self.model_id,
            )

            return self.client
        except Exception as e:
            raise Exception(f"Failed to initialize model_id: {self.model_id} with the Inference Client model.\nError: {str(e)}")

    def initialize_HFApi_client(self, temperature=0.5, timeout=120):
        """
        Initialize a client using the Hugging Face API model.

        This function initializes a client using the HfApiModel class from the smolagents library.
        It takes two optional parameters: temperature and timeout.

        Parameters:
        temperature (float): The temperature for sampling the model's output.
            Default is 0.5.
        timeout (int): The maximum time in seconds to wait for the model's response.
            Default is 120.

        Returns:
        client (HfApiModel): An instance of the HfApiModel class representing the initialized client.
            If the initialization fails, an exception is raised.

        Raises:
        Exception: If the client initialization fails.
        """
        try:
            # Initialize the client here
            self.client = HfApiModel(
                model_id=self.model_id,
                temperature=temperature,
                timeout=timeout,
            )
            return self.client
        except Exception as e:
            raise Exception(f"Failed to initialize model_id: {self.model_id} with the HF API model.\nError: {str(e)}")

    def initialize_Transformer_client(self, device="cuda"):
        """
        Initialize a model using the Local Transformer model.

        This function initializes a model using the TransformersModel class from the smolagents library.
        It takes an optional parameter: device.

        Parameters:
        device (str): The device to run the model on.
            Default is "cuda" (GPU). If "cpu" is provided, the model will run on the CPU.

        Returns:
        client (TransformersModel): An instance of the TransformersModel class representing the initialized model.
            If the initialization fails, an exception is raised.

        Raises:
        Exception: If the model initialization fails.
        """
        try:
            # Initialize the model here
            self.client = TransformersModel(
                model_id=self.model_id,
                device=device,
            )
            return self.client
        except Exception as e:
            raise Exception(f"Failed to initialize model_id: {self.model_id} with the Local Transformer model.\nError: {str(e)}")

    def initialize_LiteLLM_client(self, api_base="http://localhost:11434", api_key="YOUR_API_KEY"):
        """
        Initialize a model using the LiteLLM model.

        This function initializes a model using the LiteLLMModel class from the smolagents library.
        It takes two optional parameters: api_base and api_key.

        Parameters:
        api_base (str): The base URL of the LiteLLM API.
            Default is "http://localhost:11434".
        api_key (str): The API key for authentication.
            Default is "YOUR_API_KEY".

        Returns:
        client (LiteLLMModel): An instance of the LiteLLMModel class representing the initialized model.
            If the initialization fails, an exception is raised.

        Raises:
        Exception: If the model initialization fails.
        """
        try:
            # Initialize the model here
            self.client = LiteLLMModel(
                model_id=self.model_id,
                api_base=api_base,
                api_key=api_key,
            )
            return self.client
        except Exception as e:
            raise Exception(f"Failed to initialize model_id: {self.model_id} with the LiteLLM model.\nError: {str(e)}")
        

# def main():
#     model_init = Initialize_Client("Qwen/Qwen2.5-Coder-32B-Instruct")
#     client = model_init.initialize_Inference_client()

#     messages = [
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#     {"role": "user", "content": "What is the best time to surf?"},
#     ]
    
#     response = client.chat_completion(messages=messages)
#     print(response.choices[0].message.content)
    
    
# if __name__ == "__main__":
#     main()