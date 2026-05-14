import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "mock").lower()
        self._init_client()

    def _init_client(self):
        # Mock mode if no provider or keys are missing
        if self.provider == "mock":
            self.mock = True
            return

        self.mock = False

        if self.provider == "openai":
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                self._fallback_to_mock("OpenAI API key missing")
                return
            self.client = OpenAI(api_key=key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                self._fallback_to_mock("Anthropic API key missing")
                return
            self.client = Anthropic(api_key=key)
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        elif self.provider == "deepseek":
            from openai import OpenAI
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key:
                self._fallback_to_mock("DeepSeek API key missing")
                return
            self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        elif self.provider == "ollama":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama"
            )
            self.model = os.getenv("OLLAMA_MODEL", "llama3")

        else:
            self._fallback_to_mock(f"Unknown provider '{self.provider}'")

    def _fallback_to_mock(self, reason):
        print(f"⚠️ {reason}. Falling back to mock LLM.")
        self.mock = True

    def generate(self, prompt: str, system: str = None) -> str:
        if getattr(self, "mock", False):
            # Return a realistic mock response for thesis development
            mock_reply = f"[Mock LLM] Received prompt of length {len(prompt)}. System prompt: {system[:50] if system else 'None'}...\n\nThis is a placeholder response. Replace with real API keys when ready."
            return mock_reply

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

        else:
            return f"Unsupported provider: {self.provider}"

if __name__ == "__main__":
    llm = LLMClient()
    print(f"Using provider: {llm.provider} (mock={llm.mock})")
    print(llm.generate("What is a capability knowledge graph?"))