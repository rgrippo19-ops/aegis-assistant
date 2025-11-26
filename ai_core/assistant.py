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
  "I am Aegis, your personal assistant for AI contracting, life planning, and systems thinking."

General behavior:
- Be clear, concise, and practical.
- Think step by step for reasoning-heavy questions.
- Ask clarifying questions only when absolutely needed.
- If you do not know, say so honestly.
- Prefer structured answers with clear sections and bullet points.
- When Ryan seems overwhelmed, bias toward fewer, more realistic next steps rather than long essays.

Formatting (VERY IMPORTANT):
- Use a clear outline with line breaks, for example:

  TONIGHT'S OVERVIEW:
  - Bullet 1
  - Bullet 2

  WORKOUT:
  - Bullet 1
  - Bullet 2

  DINNER:
  - Bullet 1
  - Bullet 2

  NEXT STEPS:
  - Bullet 1
  - Bullet 2

- Each section name should be in ALL CAPS followed by a colon, on its own line.
- Each bullet must be on its own line, starting with "- ".
- Put a blank line between sections.
- Never put multiple bullets on the same line (do not write "- item 1 - item 2" on one line).
- Avoid heavy Markdown like headings or bold; use plain text with newlines.
- Keep most answers under 250–300 words unless Ryan asks you to go deep.

Modes:
- User messages may be prefixed with a tag like "[MODE: PLANNING]", "[MODE: HEALTH]" or "[MODE: MONEY]".
- The backend may also tell you which mode is active.
- Use the mode as high-priority context but still pay attention to the actual question.

Audience:
- You are primarily helping Ryan, an AI contractor who likes concrete, numeric examples and clear next actions.
- He often works on AI contracting, income planning, health routines, and long-term discipline.
""".strip()


# Extra mode-specific guidance that gets appended to the base system prompt
MODE_PROMPTS: Dict[str, str] = {
    "GENERAL": """
Mode: GENERAL
- Treat this as normal chat.
- You can range across topics but keep answers structured and action-oriented when appropriate.
""".strip(),
    "PLANNING": """
Mode: PLANNING
- Focus on structuring time, projects, and routines.
- Help Ryan break goals into weeks, days, and concrete tasks.
- Offer realistic estimates for time and effort.
- Use sections like: CONTEXT, PLAN, NEXT 3 ACTIONS.
""".strip(),
    "HEALTH": """
Mode: HEALTH
- Focus on workouts, routines, skincare, sleep, and sustainable health behaviors.
- Keep plans realistic for busy days.
- Avoid medical diagnosis; if something sounds serious, suggest consulting a medical professional.
- Use sections like: OVERVIEW, MOVEMENT, NUTRITION, SKINCARE, SLEEP, NEXT STEPS.
""".strip(),
    "MONEY": """
Mode: MONEY
- Focus on income, hours, rates, savings, and net worth planning.
- Use concrete numbers and scenarios.
- Be conservative and transparent when making assumptions.
- Use sections like: SNAPSHOT, INCOME PLAN, SAVINGS / INVESTING, RISKS, NEXT STEPS.
""".strip(),
}


@dataclass
class Tool:
    """
    Simple tool abstraction: a named function that takes a string and returns a string.
    You can extend this later if you want LLM tool calling.
    """
    name: str
    description: str
    func: Callable[[str], str]


def simple_calculator_tool(query: str) -> str:
    """
    Very simple calculator that evaluates basic arithmetic expressions.
    Expected input: something like '2 + 2 * 3'.
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
    """
    Core assistant class used by the FastAPI backend.

    - Keeps a rolling conversation history (trims old messages).
    - Parses mode tags like "[MODE: PLANNING]" from user messages.
    - Builds a system prompt that combines the base prompt with mode-specific guidance.
    - Calls the OpenAI Chat Completions API and returns the assistant's reply.
    """
    base_system_prompt: str = BASE_SYSTEM_PROMPT
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    tools: Dict[str, Tool] = field(default_factory=lambda: TOOLS)
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    history: List[Dict[str, str]] = field(default_factory=list)

    def _build_system_prompt(self, mode: Optional[str]) -> str:
        """
        Combine the base system prompt with mode-specific guidance.
        """
        if not mode:
            mode_key = "GENERAL"
        else:
            mode_key = mode.upper()
            if mode_key not in MODE_PROMPTS:
                mode_key = "GENERAL"

        mode_prompt = MODE_PROMPTS.get(mode_key, "")
        if mode_prompt:
            return f"{self.base_system_prompt}\n\n{mode_prompt}".strip()
        return self.base_system_prompt

    def _extract_mode_and_text(self, user_message: str) -> (Optional[str], str):
        """
        Parse a leading [MODE: X] tag if present and return (mode, cleaned_text).
        If no tag is present, mode is None and text is unchanged.
        """
        pattern = r"^\s*\[MODE:\s*([A-Z]+)\s*\]\s*(.*)$"
        match = re.match(pattern, user_message)
        if match:
            mode = match.group(1).upper()
            cleaned = match.group(2).strip()
            return mode, cleaned
        return None, user_message.strip()

    def _trimmed_history(self) -> List[Dict[str, str]]:
        """
        Return a trimmed copy of the history to keep context bounded.
        We assume history is a list of dicts with 'role' and 'content'.
        """
        if len(self.history) <= self.max_history_messages:
            return self.history[:]
        return self.history[-self.max_history_messages :]

    def _build_messages(self, user_message: str, mode: Optional[str]) -> List[Dict[str, str]]:
        """
        Build the full messages list for the OpenAI API:
        [system] + trimmed history + new user message.
        """
        system_prompt = self._build_system_prompt(mode)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        messages.extend(self._trimmed_history())
        messages.append({"role": "user", "content": user_message})

        return messages

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the OpenAI Chat Completions API with the given messages.
        Returns the assistant's reply as a string.
        """
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            message = response.choices[0].message.content or ""
            return message
        except Exception as e:
            # You can log e here if you add logging; for now, return a friendly error.
            return (
                "I ran into an error while trying to respond. "
                "Please try again in a moment, or adjust your message. "
                f"(Internal error: {e})"
            )

    def run_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Run one of the simple tools by name.
        """
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        return self.tools[tool_name].func(tool_input)

    def chat_step(self, user_message: str) -> str:
        """
        One step of conversation:
        - parse mode tag from the user message
        - build messages with system prompt + history + new user message
        - call LLM
        - update history
        - return assistant reply
        """
        mode, cleaned_text = self._extract_mode_and_text(user_message)

        messages = self._build_messages(cleaned_text, mode)
        reply = self.call_llm(messages)

        # Update history with cleaned text and reply
        self.add_user_message(cleaned_text)
        self.add_assistant_message(reply)

        return reply