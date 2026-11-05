from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from tools import search_runs, get_run, archive_run

SYSTEM_PROMPT = """You help ML researchers explore and manage their experiment runs.

- Use search_runs to find runs. It takes one dataset and one model per call; for several datasets or models, call it once per value.
- Use get_run when the user asks for notes, hyperparameters or full details of a specific run.
- Only call archive_run when the user explicitly asks to archive a specific run.
- Answer only from tool results. If a run doesn't exist or nothing matches, say so. Never invent runs or values.
- Keep answers short and always include run IDs."""

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model, tools=[search_runs, get_run, archive_run], system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Show runs with accuracy exactly 0.9."}]})
    for message in result["messages"]:
        message.pretty_print()