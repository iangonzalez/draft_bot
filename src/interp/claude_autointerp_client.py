from anthropic import AsyncAnthropic, Anthropic
from interp.autointerp_system_prompt import get_input_based_autointerp_system_prompt, get_output_based_autointerp_system_prompt, get_quiz_system_prompt
from enum import Enum, auto

MODEL_NAME = "claude-sonnet-4-20250514"

class AutoInterpMode(Enum):
    AUTOINTERP_ON_INPUTS = auto()
    AUTOINTERP_ON_OUTPUTS = auto() 
    AUTOINTERP_EVAL_OUTPUT_BASED_FEATURES = auto()


class ClaudeAutoInterpClient:
    def __init__(self, per_set_config, auto_interp_mode: AutoInterpMode, api_key_path: str = '/mnt/polished-lake/home/ian/anthropic_api_key.txt'):
        with open(api_key_path, 'r') as file:
            self.api_key = file.read().strip()
        if not self.api_key:
            raise ValueError("Error: ANTHROPIC_API_KEY not set.")
        
        self.system_prompt = None
        if auto_interp_mode == AutoInterpMode.AUTOINTERP_ON_INPUTS:
            self.system_prompt = get_input_based_autointerp_system_prompt(per_set_config)
        elif auto_interp_mode == AutoInterpMode.AUTOINTERP_ON_OUTPUTS:
            self.system_prompt = get_output_based_autointerp_system_prompt(per_set_config)
        elif auto_interp_mode == AutoInterpMode.AUTOINTERP_EVAL_OUTPUT_BASED_FEATURES:
            self.system_prompt = get_quiz_system_prompt()
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        self.sync_client = Anthropic(api_key=self.api_key)

    async def get_response(self, prompt_text, max_tokens=1024):
        """
        Sends a prompt to the Anthropic API and returns Claude's response using async client.

        Args:
            prompt_text (str): The user's prompt.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: Claude's response text, or None if an error occurred.
        """
        try:
            message = await self.async_client.messages.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                messages=[
                        {
                            "role": "user",
                            "content": prompt_text,
                        }
                    ],
                    system=self.system_prompt
                )
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        return message.content[0].text
    
    def start_batch_prompt_processing(self, prompts: list[str], max_tokens: int = 1024):
        """
        Takes a list of prompts and submits them to the API for batch processing.
        
        Args:
            prompts (list[str]): List of prompts to send to the API
            max_tokens (int): Maximum number of tokens to generate per response
            
        Returns:
            BetaMessagesBatch: The batch information.
        """
        batch_requests = []
        for i, prompt in enumerate(prompts):
            request = {
                "params" : {
                    "model": MODEL_NAME,
                    "max_tokens": max_tokens, 
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "system": self.system_prompt,
                },
                "custom_id": f"msgbatch_{i}"
            }
            batch_requests.append(request)
        batch_info = self.sync_client.messages.batches.create(requests=batch_requests)
        return batch_info
    