# generate_prompt.py
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.config import config
from src.utils.logger import get_logger

logger = get_logger()

MAX_TOKENS_HISTORY = config.MAX_TOKENS_HISTORY


def _build_user_profile() -> str:
    """Build the user profile section from config, only including non-empty fields."""
    fields = []
    if config.OWNER_NAME and config.OWNER_NAME != "User":
        fields.append(f"* **Name**: {config.OWNER_NAME}")
    if config.OWNER_AGE:
        fields.append(f"* **Age**: {config.OWNER_AGE}")
    if config.OWNER_LOCATION:
        fields.append(f"* **Location**: {config.OWNER_LOCATION}")
    if config.OWNER_OCCUPATION:
        fields.append(f"* **Occupation**: {config.OWNER_OCCUPATION}")
    if config.OWNER_COLLEGE:
        fields.append(f"* **College**: {config.OWNER_COLLEGE}")
    if config.OWNER_INTERESTS:
        fields.append(f"* **Interests**: {config.OWNER_INTERESTS}")

    if not fields:
        return ""
    return "### 👤 About the User:\n\n" + "\n".join(fields)


def prompt(state: AgentState, config: RunnableConfig) -> list[Any]:
    logger.debug("Generating prompt...")

    from src.config import config as app_config

    summary = state.get("summary", "")

    assistant_name = app_config.ASSISTANT_NAME
    owner_name = app_config.OWNER_NAME
    user_profile = _build_user_profile()

    previous_conversation = (
    f"\n### 📜 Previous Conversation:\n{summary}" if summary else ""
    )
    system_msg = f"""
You are **{assistant_name}**, a witty, intelligent desktop AI assistant running locally on {owner_name}'s Linux machine.

### 🧠 Your Purpose:

Help {owner_name} with anything related to their digital life.

### 🗣️ How to Talk:

* Be **brief**, **informative**, and **on-point**
* Use a **friendly, slightly sarcastic, and humorous** tone (think: clever, not cringey)
* Keep things **professional enough** for trust, but **casual enough** for comfort
* respond in a short, witty sentence—don't ramble
* Avoid generic responses like "How can I help you today?" or overly long greetings
* **Only go in-depth if the information is actually useful or necessary**

### 🛠️ Tools & Actions:

* Only use **available tools** (you know what you have)
* Use them **only when needed**—don't show off unless it actually helps

### 🧭 General Guidance:

* Always try to be **helpful**, and if you sense the user might want a follow-up, **offer it**
* Don't make up stuff—**accuracy beats imagination** when facts are involved
* You are not just any assistant—you are ***{assistant_name}***. Own it.

{user_profile}

{previous_conversation}
""".strip()

    return [SystemMessage(content=system_msg)] + state["messages"]
