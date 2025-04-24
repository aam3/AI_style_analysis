import os
import anthropic
from typing import Optional, Dict, Any

class TextGenerator:
    def __init__(self, api_key=None, model_name="claude-3-7-sonnet-20250219"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = model_name
        self.client = self._init_client()
        
    def _init_client(self):
        """Initialize the Claude API client"""
        return anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt_template, system_setting, **prompt_params):
        """Generate text using the provided prompt template and parameters"""
        formatted_prompt = prompt_template.format(**prompt_params)
        
        response = self.client.messages.create(
            model=self.model_name,
            temperature=1.0,
            max_tokens=10000,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        
        return response.content[0].text