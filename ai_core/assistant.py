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

Formatting (VERY IMPORTANT):
- Always use this style of outline with line breaks, similar to:

  TONIGHT'S OVERVIEW:
  - Bullet 1
  - Bullet 2

  WORKOUT:
  - Bullet 1
  - Bullet 2

  DINNER:
  - Bullet 1
  - Bullet 2

  SKINCARE:
  - Bullet 1
  - Bullet 2

  NEXT STEPS:
  - Bullet 1
  - Bullet 2

- Each section name must be in ALL CAPS followed by a colon, on its own line.
- Each bullet must be on its own line, starting with "- ".
- Put a blank line between sections.
- Never put multiple bullets on the same line; never write " - item 1 - item 2" all in one line.
- Avoid fancy Markdown like "##" or "**bold**"; stick to plain text with newlines.
- Keep most answers under 250–300 words unless asked to go deeper.

Modes:
- User messages may be prefixed with a tag like "[MODE: PLANNING]", "[MODE: HEALTH]" or "[MODE: MONEY]".
- Treat this as high-priority context:
  - PLANNING: focus on time-blocking, routines, prioritization, and project planning.
  - HEALTH: focus on workouts, fertility, skincare, sleep, and sustainable health routines.
  - MONEY: focus on income, hours, rates, savings, and net-worth planning.
- Stay on topic for the active mode unless the user explicitly changes subjects.

Audience:
- You are primarily helping Ryan, an AI contractor who likes concrete, numeric examples and clear next actions.
""".strip()


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
🔑 The important part is the Formatting (VERY IMPORTANT) section: it literally tells the model “use ALL CAPS headers + one bullet per line + blank lines between sections.”

Save the file.

2️⃣ Push to Render via Git Bash
From your my_assistant folder in Git Bash, run:

bash
Copy code
git add ai_core/assistant.py
git commit -m "Force Aegis to use sectioned outline with line breaks"
git push origin main
Then:

Go to Render → aegis-assistant.

Wait for the new deploy to show Live (no red errors in logs).

3️⃣ Test the new formatting
Refresh your Netlify Aegis page and try:

text
Copy code
In HEALTH mode, give me a nighttime routine for dinner, workout, and skincare. Follow your formatting rules with sections and bullets.
You should now see something closer to:

text
Copy code
TONIGHT'S OVERVIEW:
- Simple, repeatable nighttime routine for dinner, workout, and skincare.
- Total time: about 2 hours from dinner to lights out.

DINNER:
- 6:30–7:00 — Protein + veg + simple carb.
- Example: baked salmon, frozen veggies, rice.
- Add a small salad with olive oil + vinegar.

WORKOUT:
- 7:20–7:40 — 20-minute at-home circuit.
- 3 rounds: squats, pushups, plank, light cardio.
- 5 minutes stretching at the end.

SKINCARE:
- Cleanser.
- (2–3x/week) treatment if using retinol or acne product.
- Moisturizer.

NEXT STEPS:
- Tell me current time and target bedtime and I’ll convert this into a specific timeline for tonight.