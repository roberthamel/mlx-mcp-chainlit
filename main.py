import os
from openai import AsyncOpenAI
import chainlit as cl
from typing import Dict, Any, List
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

model = os.getenv("MODEL", "qwen/qwen3-8b")
client = AsyncOpenAI(base_url="http://localhost:13333/v1", api_key="mlx")
cl.instrument_openai()

settings = {
    "model": model,
    "temperature": 0,
    "stream": True,
}

mcp_tools_cache = {}


@cl.on_chat_start
async def start():
    cl.user_session.set(
        "message_history",
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that uses local mlx models and MCP servers. When you need to use a tool, call the appropriate function with the required parameters in your response.",
            }
        ],
    )

    await cl.Message(
        content="Welcome! I'm using a local model running in mlx with MCP integration. Make sure that: \n"
        "1. mlx is running \n"
        "2. A default model is loaded \n"
        "3. The mlx server has started \n"
    ).send()


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    await cl.Message(f"Connected to MCP server: {connection.name}").send()

    try:
        result = await session.list_tools()

        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

        mcp_tools_cache[connection.name] = tools

        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)

        await cl.Message(
            f"Found {len(tools)} tools from {connection.name} MCP server."
        ).send()
    except Exception as e:
        await cl.Message(f"Error listing tools from MCP server: {str(e)}").send()


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    if name in mcp_tools_cache:
        del mcp_tools_cache[name]

    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

    await cl.Message(f"Disconnected from MCP server: {name}").send()


@cl.step(type="tool")
async def execute_tool(tool_name: str, tool_input: Dict[str, Any]):
    print("Executing tool:", tool_name)
    print("Tool input:", tool_input)
    mcp_name = None
    mcp_tools = cl.user_session.get("mcp_tools", {})

    for conn_name, tools in mcp_tools.items():
        if any(tool["name"] == tool_name for tool in tools):
            mcp_name = conn_name
            break

    if not mcp_name:
        return {"error": f"Tool '{tool_name}' not found in any connected MCP server"}

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    try:
        result = await mcp_session.call_tool(tool_name, tool_input)
        return result
    except Exception as e:
        return {"error": f"Error calling tool '{tool_name}': {str(e)}"}


async def format_tools_for_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    openai_tools = []

    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def format_calltoolresult_content(result):
    """Extract text content from a CallToolResult object.

    The MCP CallToolResult contains a list of content items,
    where we want to extract text from TextContent type items.
    """
    text_contents = []

    if isinstance(result, CallToolResult):
        for content_item in result.content:
            # This script only supports TextContent but you can implement other CallToolResult types
            if isinstance(content_item, TextContent):
                text_contents.append(content_item.text)

    if text_contents:
        return "\n".join(text_contents)
    return str(result)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    try:
        # Initial message for the first assistant response
        initial_msg = cl.Message(content="")
        await initial_msg.send()

        mcp_tools = cl.user_session.get("mcp_tools", {})
        all_tools = []
        for connection_tools in mcp_tools.values():
            all_tools.extend(connection_tools)

        chat_params = {**settings}
        if all_tools:
            openai_tools = await format_tools_for_openai(all_tools)
            chat_params["tools"] = openai_tools
            chat_params["tool_choice"] = "auto"
            print("Tools passed:", openai_tools)
        stream = await client.chat.completions.create(
            messages=message_history, **chat_params
        )

        initial_response = ""
        tool_calls = []

        async for chunk in stream:
            delta = chunk.choices[0].delta
            print(delta)

            if token := delta.content or "":
                initial_response += token
                await initial_msg.stream_token(token)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc_id = tool_call.index
                    if tc_id >= len(tool_calls):
                        tool_calls.append({"name": "", "arguments": ""})

                    if tool_call.function.name:
                        tool_calls[tc_id]["name"] = tool_call.function.name

                    if tool_call.function.arguments:
                        tool_calls[tc_id]["arguments"] += tool_call.function.arguments

        # First, update message history with the initial response
        if initial_response.strip():
            message_history.append({"role": "assistant", "content": initial_response})

        # Process tool calls if any
        if not tool_calls and initial_response.strip():
            # Try to parse <tool_call>...</tool_call> blocks from initial_response
            import re
            import json

            tool_call_matches = re.findall(
                r"<tool_call>\s*(.*?)\s*</tool_call>", initial_response, re.DOTALL
            )
            for match in tool_call_matches:
                try:
                    tc = json.loads(match)
                    tool_calls.append(
                        {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
                    )
                except Exception:
                    continue

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                try:
                    import json

                    tool_args = json.loads(tool_call["arguments"])
                    message_history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{len(message_history)}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call["arguments"],
                                    },
                                }
                            ],
                        }
                    )
                    with cl.Step(name=f"Executing tool: {tool_name}", type="tool"):
                        tool_result = await execute_tool(tool_name, tool_args)
                    tool_result_content = format_calltoolresult_content(tool_result)
                    tool_result_msg = cl.Message(
                        content=f"Tool Result from {tool_name}:\n{tool_result_content}",
                        author="Tool",
                    )
                    await tool_result_msg.send()
                    message_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{len(message_history) - 1}",
                            "content": tool_result_content,
                        }
                    )
                    follow_up_msg = cl.Message(content="")
                    await follow_up_msg.send()
                    follow_up_stream = await client.chat.completions.create(
                        messages=message_history, **settings
                    )
                    follow_up_text = ""
                    async for chunk in follow_up_stream:
                        if token := chunk.choices[0].delta.content or "":
                            follow_up_text += token
                            await follow_up_msg.stream_token(token)
                    message_history.append(
                        {"role": "assistant", "content": follow_up_text}
                    )
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    error_message = cl.Message(content=error_msg)
                    await error_message.send()

        # Update the session message history
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        await cl.Message(content=error_message).send()

        troubleshooting = (
            "Troubleshooting tips:\n"
            "1. Verify mlx is running\n"
            "2. Check that a model is loaded\n"
            "3. Confirm the mlx server is started on port 1234\n"
            "4. Make sure the model supports the OpenAI chat completions API format with tools"
        )
        await cl.Message(content=troubleshooting).send()


if __name__ == "__main__":
    import sys
    import signal
    import chainlit.cli

    def sigint_handler(*_):
        print("\nShutting down gracefully (Ctrl+C pressed).")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    try:
        chainlit.cli.run(sys.argv[1:] or ["run", "main.py"])
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down gracefully (Ctrl+C pressed).")
        sys.exit(0)
