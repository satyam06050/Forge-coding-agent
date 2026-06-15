from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated
from langchain_mcp_adapters.tools import load_mcp_tools

import asyncio
import logging
import os

logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

load_dotenv()

WINDOW_SIZE = 10


class conversation_history(TypedDict):
    folder_structure_path: str
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

servers = {
    "server": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            os.path.join(PROJECT_DIR, "server.py")
        ]
    }
}

folder_path = PROJECT_DIR

system = SystemMessage(content=f"""You are an expert AI coding assistant. You help users read, write,
refactor, debug, and execute code inside their project.

PROJECT ROOT: {folder_path}
Always use this as the base to determine each file path.

## How to behave

- Use the MINIMUM number of tool calls possible. Think before calling a tool.
- If the user asks about a specific file by name, call read_file with PROJECT_ROOT/filename directly.
- If read_file fails (file not found), call search_in_files ONCE with the filename as pattern and PROJECT_ROOT as directory. Then read the exact path it returns. Do NOT explore folders or run terminal commands to find it.
- Before editing any file, read it first.
- For small edits use find_and_replace_in_file. Use write_file only to rewrite the whole file.
- Before writing NEW code where the file location is unknown, use list_all_files_in_project ONCE.
- When asked to add or change something, make targeted edits — do not rewrite the entire file unless necessary.
- Always write clean, idiomatic code that matches the style of the existing codebase.
- If a task is ambiguous, make a reasonable assumption and state it clearly.

## When to use tools

USE read_file when:
- The user asks what a specific file does or contains — read it directly, nothing else.

USE find_and_replace_in_file for targeted edits (changing a specific line or block).
USE write_file only when you need to rewrite the entire file.

USE search_in_files when:
- You need to locate a file by name after read_file fails.
- Looking for where a function, class, or variable is defined before editing it.

USE list_all_files_in_project only when:
- You need to understand the full project layout before writing new code.

USE run_terminal_command when:
- The user explicitly asks you to run, execute, or test something.
- You write new code and need to verify it actually works.
- Installing a dependency is required before code will run.

DO NOT run terminal commands when:
- The task is purely about reading, explaining, or writing files.
- No new logic was introduced that could break things.

USE git tools when:
- The user asks to commit, check status, or see what changed.

## Response style

- Be concise. Don't over-explain obvious things.
- After completing a task, give a short summary of what you did and what files changed.
- If you ran a command, always include the output.
- If something failed, explain why and attempt to fix it automatically.
- Never ask clarifying questions mid-task — make your best judgment and proceed.
""")


def split_into_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Split message list into complete turns, each starting with a HumanMessage."""
    turns = []
    current_turn = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and current_turn:
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)
    return turns


async def summarize_turns(turns: list[list[BaseMessage]], llm) -> str:
    """Summarize old turns into a concise string, skipping tool messages."""
    lines = []
    for turn in turns:
        human = turn[0].content if turn else ""
        ai_reply = ""
        for msg in reversed(turn):
            if isinstance(msg, AIMessage):
                c = msg.content
                if isinstance(c, list):
                    ai_reply = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                else:
                    ai_reply = str(c)
                break
        lines.append(f"Human: {human}\nAssistant: {ai_reply}")

    prompt = f"""Summarize the following conversation turns concisely.
Preserve key decisions, file names, code changes, and any important context for future reference.

{chr(10).join(lines)}

Summary:"""
    response = await llm.ainvoke(prompt)
    content = response.content
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content if isinstance(p, dict))


async def build_graph():
    client = MultiServerMCPClient(servers)
    async with client.session("server") as session:
        tools = await load_mcp_tools(session)
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.5)
        llm_with_tools = llm.bind_tools(tools)
        tool_dict = {t.name: t for t in tools}

        async def chat_node(state: conversation_history) -> conversation_history:
            all_messages = list(state["messages"])
            existing_summary = state.get("summary", "")
            new_summary = existing_summary

            turns = split_into_turns(all_messages)
            if len(turns) > WINDOW_SIZE:
                old_turns = turns[:-WINDOW_SIZE]
                recent_turns = turns[-WINDOW_SIZE:]
                summarized_text = await summarize_turns(old_turns, llm)
                new_summary = f"{existing_summary}\n\n{summarized_text}".strip() if existing_summary else summarized_text
                recent_messages = [msg for turn in recent_turns for msg in turn]
            else:
                recent_messages = all_messages

            context = [system]
            if new_summary:
                context.append(SystemMessage(content=f"Summary of earlier conversation:\n{new_summary}"))
            context.extend(recent_messages)

            llm_response = await llm_with_tools.ainvoke(context)
            new_messages = [llm_response]

            while llm_response.tool_calls:
                for i in llm_response.tool_calls:
                    tool_to_call = tool_dict[i["name"]]
                    response_by_tool = await tool_to_call.ainvoke(i["args"])
                    new_messages.append(ToolMessage(content=response_by_tool[0]["text"], tool_call_id=i["id"]))
                llm_response = await llm_with_tools.ainvoke(context + new_messages)
                new_messages.append(llm_response)

            return {"messages": new_messages, "summary": new_summary}

        graph = StateGraph(conversation_history)
        graph.add_node("chat_node", chat_node)
        graph.add_edge(START, "chat_node")
        graph.add_edge("chat_node", END)

        config = {"configurable": {"thread_id": "thread_1"}}
        async with AsyncSqliteSaver.from_conn_string("chatbot.db") as checkpointer:
            CompiledGraph = graph.compile(checkpointer=checkpointer)
            while True:
                question = input("query : ")
                if question == "exit":
                    break
                response = await CompiledGraph.ainvoke(
                    {"folder_structure_path": folder_path, "messages": [HumanMessage(content=question)]},
                    config=config
                )
                last_message = response["messages"][-1]
                content = last_message.content
                if isinstance(content, list):
                    content = " ".join(c["text"] for c in content if isinstance(c, dict) and "text" in c)
                print("\nAgent:", content)


async def main():
    await build_graph()


if __name__ == "__main__":
    asyncio.run(main())
