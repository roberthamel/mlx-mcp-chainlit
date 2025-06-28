import json
import time
import chainlit as cl
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from openai import AsyncOpenAI


client = AsyncOpenAI(base_url="http://localhost:13333/v1", api_key="mlx")
cl.instrument_openai()


settings = {
    "model": "Qwen/Qwen3-8B",
    "temperature": 0,
    "stream": True,
}


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    cl.user_session.set(
        "mcp_tools",
        {},
    )
    cl.Message(
        content="Welcome to the Qwen3 model chat! What would you like to discuss?"
    )


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    await cl.Message(f"Connected to MCP server: {connection.name}").send()

    try:
        result = await session.list_tools()

        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_scheme": t.inputSchema,
            }
            for t in result.tools
        ]

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
    mcp_tools = cl.user_session.get("mcp_tools", {})

    if name in mcp_tools:
        del mcp_tools[name]
        cl.user_session.set("mcp_tools", mcp_tools)

    await cl.Message(f"Disconnected from MCP server: {name}").send()


@cl.step(type="tool")
async def execute_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input

    current_step = cl.context.current_step
    current_step.name = tool_name

    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None

    for conn_name, tools in mcp_tools.items():
        if any(tool["name"] == tool_name for tool in tools):
            mcp_name = conn_name
            break

    if not mcp_name:
        current_step.output = json.dumps(
            {"error": f"Tool '{tool_name}' not found in any connected MCP server"}
        )
        return current_step.output

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    if not mcp_session:
        current_step.output = json.dumps(
            {"error": f"MCP {mcp_name} not found in any MCP connection"}
        )
        return current_step.output

    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps(
            {"error": f"MCP {mcp_name} not found in any MCP connection: {str(e)}"}
        )
    return current_step.output


async def format_tools_for_openai(tools):
    openai_tools = []

    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_scheme"],
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def format_calltoolresult_content(result):
    text_contents = ""

    if isinstance(result, CallToolResult):
        for content_item in result.content:
            if isinstance(content_item, TextContent):
                text_contents.append(content_item.text)

    if text_contents:
        return "\n".join(text_contents)

    return str(result)


@cl.on_message
async def on_message(message: cl.Message):
    start = time.time()

    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    try:
        initial_msg = cl.Message(content="")
        mcp_tools = cl.user_session.get("mcp_tools", {})

        all_tools = []

        for connection_tools in mcp_tools.values():
            all_tools.extend(connection_tools)

        chat_params = {**settings}

        if all_tools:
            openai_tools = await format_tools_for_openai(all_tools)
            chat_params["tools"] = openai_tools
            chat_params["tool_choice"] = "auto"
            print("Tools passed", openai_tools)

        stream = await client.chat.completions.create(
            messages=message_history, **chat_params
        )

        thinking = False

        async with cl.Step("Thinking") as thinking_step:
            final_answer = cl.Message(content="")

            async for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content == "<think>":
                    thinking = True
                    continue

                if delta.content == "</think>":
                    thinking = False
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    continue

                if thinking:
                    await thinking_step.stream_token(delta.content)
                else:
                    await final_answer.stream_token(delta.content)

        initial_response = ""
        tool_calls = []

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if token := delta.content or "":
                initial_response += token
                await initial_msg.stream_token(token)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tc_id = tool_call.index

                    if tc_id >= len(tool_calls):
                        tool_calls.append({"name": "", "arguments": ""})

                    if tool_call.function_name:
                        tool_calls[tc_id]["name"] = tool_call.function_name

                    if tool_call.function.arguments:
                        tool_calls[tc_id]["arguments"] = tool_call.function.arguments

            if initial_response.strip():
                message_history.append(
                    {"role": "assistant", "content": initial_response}
                )

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
                        content=f"Tool result from {tool_name}:\n{tool_result_content}",
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
                    follow_up_stream = await client.chat.completions.create(
                        messages=message_history, **settings
                    )

                    follow_up_text = ""
                    async for chunk in follow_up_stream:
                        if token := chunk.choice[0].delta.content or "":
                            follow_up_text += token
                            await follow_up_msg.stream_token(token)

                    message_history.append(
                        {"role": "assistant", "content": follow_up_text}
                    )

                except Exception as e:
                    error_message = cl.Message(
                        content=f"Error executing tool {tool_name}: {str(e)}"
                    )
                    await error_message.send()

        cl.user_session.set("message_history", message_history)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        await cl.Message(content=error_message).send()
