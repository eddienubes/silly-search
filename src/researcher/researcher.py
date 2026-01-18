import asyncio
import typing
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command
import state
from ..config import cfg
from . import tools
from .. import tools as common_tools
from .. import prompts
from .. import utils


async def research(
    state: state.ResearcherState,
) -> Command[typing.Literal["handle_researcher_tools"]]:

    available_tools = [
        tools.search,
        common_tools.think,
        common_tools.ResearchCompleteTool,
    ]
    llm = (
        init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
        .bind_tools(available_tools)
        .with_retry(stop_after_attempt=cfg.max_llm_retries)
    )

    system_prompt = prompts.researcher_system_prompt.format(
        data=utils.get_readable_date(),
        mcp_prompt="",
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.get("research_topic")),
    ]

    response = await llm.ainvoke(messages)

    return Command(
        goto="handle_researcher_tools",
        update={
            "researcher_messsages": messages + [response],
            "researcher_iterations": state.get("researcher_iterations", 0) + 1,
        },
    )


async def handle_researcher_tools(
    state: state.ResearcherState,
) -> Command[typing.Literal["researcher", "compress_research"]]:
    latest_message = state.get("researcher_messages")[-1]

    latest_message = typing.cast(AIMessage, latest_message)
    has_finished = any(
        call
        for call in latest_message.tool_calls
        if call["name"] == common_tools.ResearchCompleteTool.__name__
    )

    tool_calls = [
        call
        for call in latest_message.tool_calls
        if call["name"] in [tools.search.name, common_tools.think.name]
    ]

    tool_call_tasks = [
        utils.run_safe(
            tools.search.ainvoke,
            msg=f"Error when caling a tool {call['name']}",
            **call["args"],
        )
        for call in tool_calls
    ]

    awaited_tool_call_tasks = await asyncio.gather(*tool_call_tasks)

    tool_outputs = [
        ToolMessage(content=result, tool_call_id=call["id"], name=call["name"])
        for result, call in zip(awaited_tool_call_tasks, tool_calls)
    ]

    has_exceeded_max_calls = (
        state.get("researcher_iterations") > cfg.max_researcher_iterations
    )
    if has_finished or has_exceeded_max_calls:
        return Command(
            goto="compress_research", update={"researcher_messages": [tool_outputs]}
        )

    return Command(goto="researcher", update={"researcher_messages": [tool_outputs]})


async def compress_research(
    state: state.ResearcherState,
) -> Command[typing.Literal["__end__"]]:
    llm = init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key).with_retry(
        stop_after_attempt=cfg.max_llm_retries
    )

    system_prompt = prompts.research_compressor_system_prompt
    human_prompt = prompts.research_compressor_human_prompt

    messages = [
        SystemMessage(content=system_prompt),
        *state.get("researcher_messages"),
        HumanMessage(content=human_prompt),
    ]

    response = await llm.ainvoke(messages)

    return Command(
        goto="__end__", update={"compressed_research": str(response.content)}
    )
