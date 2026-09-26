# assistant.py
import base64
import sqlite3

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.config import config
from src.core.llm import model
from src.utils.logger import get_logger

from .generate_prompt import prompt
from .summarizer import summarize_conversation
from .tools import get_all_tools  # Import the function to get all tools

logger = get_logger()

conn = sqlite3.connect(config.CHECKPOINTS_DB, check_same_thread=False)
checkpointer = SqliteSaver(conn)


class State(AgentState):
    summary: str


logger.info("Chat model initialized.")

agent = create_react_agent(
    model=model,
    tools=get_all_tools(),
    prompt=prompt,
    state_schema=State,
    checkpointer=checkpointer,
)

logger.info("Agent initialized.")


def call_agent(message):
    config_dict = {"configurable": {"thread_id": config.THREAD_ID}}
    state = agent.get_state(config_dict)
    messages = state.values.get("messages", [])
    token_count = count_tokens_approximately(messages)

    if token_count > config.MAX_TOKENS_HISTORY:
        summary_result = summarize_conversation(
            {
                "messages": messages,
                "summary": state.values.get("summary", ""),
            }
        )
        agent.update_state(
            config_dict,
            summary_result,
        )
    response = agent.invoke(
        {"messages": [message]},
        config_dict,
    )
    msg = response["messages"][-1].content

    # Handle the case when response["messages"][-1].content is a list of messages
    if isinstance(msg, list):
        msg = "\n".join([msg.content for msg in msg])
    if msg.startswith("tool_message:"):
        tool_name = msg.split(":")[1]
        tool_args = msg.split(":")[2]
        if tool_name == "get_screenshot":
            with open(tool_args, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                                }
                            ],
                        }
                    ]
                },
                config=config_dict,
            )

            msg = response["messages"][-1].content

            # Handle the case when response["messages"][-1].content is a list of messages
            if isinstance(msg, list):
                msg = "\n".join([msg.content for msg in msg])
            return msg
        return msg
    return msg


if __name__ == "__main__":
    # Run the agent
    message = HumanMessage(content="Hey, how are you doing?")

    response = call_agent(message)

    print(response)
