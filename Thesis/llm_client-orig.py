import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        elif self.provider == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        elif self.provider == "qwen":
            from dashscope import Generation
            self.client = Generation
            self.model = os.getenv("QWEN_MODEL", "qwen-turbo")
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
            self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        elif self.provider == "ollama":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama"
            )
            self.model = os.getenv("OLLAMA_MODEL", "llama3")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str, system: str = None) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.provider in ("openai", "deepseek", "ollama"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        elif self.provider == "qwen":
            from dashscope import Generation
            response = self.client.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.output.text
        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)
            return response.text
        else:
            raise NotImplementedError

if __name__ == "__main__":
    llm = LLMClient()
    print(f"Using provider: {llm.provider}")
    print(llm.generate("What is a capability knowledge graph?"))