from dataclasses import dataclass, field
from typing import List, Dict, Callable

from openai import OpenAI

# Client uses OPENAI_API_KEY from the environment
client = OpenAI()

DEFAULT_MODEL = "gpt-5.1"
DEFAULT_TEMPERATURE = 0.95

BASE_SYSTEM_PROMPT = """
You are Ryan's personal AI assistant named Aegis.

Identity:
- You do not say you are ChatGPT or an OpenAI model unless explicitly asked.
- When asked who you are, say something like:
  "I am Aegis, your personal assistant for AI contracting, life planning, and systems thinking. I also use ChatGPT 5.1"

General behavior:
- Be clear, concise, and practical.
- Think step by step for reasoning-heavy questions.
- Ask clarifying questions only when absolutely needed.
- If you do not know, say so honestly.
- Prefer structured answers: short paragraphs, bullet points, checklists.
- Provide a combination of sycophantic respones and sarcarstic roasts


Specialization:
- You help with:
  - AI contract work (hours, rates, planning, skill development)
  - Income and net worth planning
  - Health routines
  - Long-term habits and discipline
  - Turning messy ideas into structured plans and sprints
  - Providing emotional support 
  

Tone:
- Supportive but direct.
- Use concrete numbers and examples when talking about income and hours.
- If Ryan seems overwhelmed, suggest smaller steps and prioritization.
""".strip()

<style>
  body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
    background: #f5f5f7;
  }

  h1 {
    margin-bottom: 0.25rem;
  }

  p {
    margin-top: 0;
    color: #555;
  }

  #messages {
    border: 1px solid #ddd;
    padding: 1rem;
    height: 420px;
    overflow-y: auto;
    background: #fafafa;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    gap: 0.75rem; /* space between bubbles */
  }

  .msg-user,
  .msg-assistant {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    line-height: 1.5;
    white-space: pre-wrap; /* 🔑 preserves line breaks from the model */
    font-size: 0.95rem;
  }

  .msg-user {
    background: #e6f2ff;
    align-self: flex-end;
  }

  .msg-assistant {
    background: #f1f1f1;
    align-self: flex-start;
  }

  #input-row {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
  }

  #message {
    flex: 1;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    border: 1px solid #ccc;
    font-size: 0.95rem;
  }

  #send {
    padding: 0.5rem 0.9rem;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-size: 0.95rem;
  }

  #send:hover {
    filter: brightness(0.97);
  }
</style>


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]  # takes a string input, returns string output


def simple_calculator_tool(query: str) -> str:
    """
    Very simple calculator that evaluates basic arithmetic expressions.
    Expected input: something like '2 + 2 * 3'
    """
    import math
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}

    try:
        code = compile(query, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise ValueError(f"Use of '{name}' not allowed in calculator.")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


TOOLS: Dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description="Evaluate a math expression, e.g. '2 + 2 * 3'.",
        func=simple_calculator_tool,
    ),
}


@dataclass
class ChatAssistant:
    system_prompt: str = BASE_SYSTEM_PROMPT
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    tools: Dict[str, Tool] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        # Initialize conversation with system message
        self.history.insert(0, {"role": "system", "content": self.system_prompt})

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def call_llm(self) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
        )
        message = response.choices[0].message.content
        return message

    def run_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        return self.tools[tool_name].func(tool_input)

    def chat_step(self, user_message: str) -> str:
        """
        One step of conversation:
        - add user message
        - call model
        - return assistant reply
        """
        self.add_user_message(user_message)
        reply = self.call_llm()
        self.add_assistant_message(reply)
        return reply
