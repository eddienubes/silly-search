import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from config import cfg
import supervisor.supervisor_state as supervisor_state
import prompts
import utils
import typing
import common_tools
import supervisor.supervisor_tools as supervisor_tools


class ClarifyUserRequestOutputSchema(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question"
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope"
    )
    verification: str = Field(
        description="Verify message that we will start the research after the user has provided all the necessary information"
    )


async def clarify_user_request(
    state: supervisor_state.SillySearchState,
) -> Command[typing.Literal["write_research_brief", "__end__"]]:
    llm = init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
    model = llm.with_structured_output(ClarifyUserRequestOutputSchema).with_retry(
        stop_after_attempt=cfg.max_llm_retries
    )

    result = await model.ainvoke(
        [
            HumanMessage(
                content=prompts.clarify_prompt.format(
                    messages=get_buffer_string(state["messages"]),
                    date=utils.get_readable_date(),
                ),
            )
        ],
    )

    result = typing.cast(ClarifyUserRequestOutputSchema, result)

    if result.need_clarification:
        return Command(
            update={"messages": [AIMessage(content=result.question)]}, goto="__end__"
        )

    return Command(
        update={"messages": [AIMessage(content=result.verification)]},
        goto="write_research_brief",
    )


class ResearchBriefOutputSchema(BaseModel):
    research_brief: str = Field(
        description="A research question that will be used to guide the research"
    )


async def write_research_brief(
    state: supervisor_state.SillySearchState, config: RunnableConfig
) -> Command[typing.Literal["supervise"]]:
    llm = init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
    model = llm.with_structured_output(ResearchBriefOutputSchema).with_retry(
        stop_after_attempt=cfg.max_llm_retries
    )

    result = await model.ainvoke(
        [
            HumanMessage(
                content=prompts.create_research_brief_prompt.format(
                    messages=get_buffer_string(state["messages"]),
                    date=utils.get_readable_date(),
                )
            )
        ]
    )

    result = typing.cast(ResearchBriefOutputSchema, result)

    return Command(
        update={
            "messages": [AIMessage(content=result.research_brief)],
            "research_brief": result.research_brief,
        },
        goto="supervise",
    )


async def supervise(
    state: supervisor_state.SillySearchState, runtime: Runtime
) -> Command[typing.Literal["handle_supervisor_tools"]]:
    available_tools = [
        common_tools.ResearchCompleteTool,
        common_tools.think,
        supervisor_tools.invoke_researcher,
    ]

    llm = (
        init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
        .bind_tools(available_tools)
        .with_retry(stop_after_attempt=cfg.max_llm_retries)
    )

    brief = state.get("research_brief")
    system_prompt = prompts.supervisor_prompt.format(
        date=utils.get_readable_date(),
        max_researcher_iterations=cfg.max_supervisor_iterations,
        max_concurrent_research_units=cfg.max_concurrent_researchers,
    )

    messages = []
    supervisor_messages = state.get("supervisor_messages")

    if supervisor_messages:
        messages = supervisor_messages
    else:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=brief)]

    response = await llm.ainvoke(messages)

    messages += [response]

    return Command(
        goto="handle_supervisor_tools",
        update={
            "supervisor_messages": messages,
            "supervisor_iterations": state.get("supervisor_iterations", 0) + 1,
        },
    )


async def handle_supervisor_tools(
    state: supervisor_state.SillySearchState, researcher: CompiledStateGraph
) -> Command[typing.Literal["__end__", "supervise"]]:
    latest_message = state.get("supervisor_messages")[-1]
    latest_message = typing.cast(AIMessage, latest_message)

    has_exceeded_supervisor_iterations = (
        state.get("supervisor_iterations", 0) > cfg.max_supervisor_iterations
    )
    has_no_tool_calls = not latest_message.tool_calls
    is_research_complete = any(
        call["name"] == common_tools.ResearchCompleteTool.__name__
        for call in latest_message.tool_calls
    )

    if is_research_complete or has_exceeded_supervisor_iterations or has_no_tool_calls:
        notes = [
            call.content
            for call in filter_messages(
                state.get("supervisor_messages", []), include_types="tool"
            )
        ]
        return Command(goto="__end__", update={"notes": notes})

    think_tool_calls = [
        call
        for call in latest_message.tool_calls
        if call["name"] == common_tools.think.name
    ]

    messages = []

    for call in think_tool_calls:
        content = await common_tools.think.ainvoke(call)
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    researcher_tool_calls = [
        call
        for call in latest_message.tool_calls
        if call["name"] == supervisor_tools.invoke_researcher.name
    ]

    if researcher_tool_calls:
        allowed_calls_tool_calls = researcher_tool_calls[
            : cfg.max_concurrent_researchers
        ]
        overflow_tool_calls = researcher_tool_calls[cfg.max_concurrent_researchers :]

        researcher_tasks = [
            supervisor_tools.invoke_researcher.ainvoke(
                {**call["args"], "researcher": researcher}
            )
            for call in researcher_tool_calls
        ]

        search_hits = await asyncio.gather(*researcher_tasks)

        for call in overflow_tool_calls:
            messages.append(
                ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {cfg.max_concurrent_researchers} or fewer research units.",
                    tool_call_id=call["id"],
                )
            )

        for call, hit in zip(allowed_calls_tool_calls, search_hits):
            messages.append(ToolMessage(content=hit, tool_call_id=call["id"]))

        return Command(goto="supervise", update={"supervisor_messages": messages})

    return Command(goto="supervise", update={"supervisor_messages": messages})
