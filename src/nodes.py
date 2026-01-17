from typing import Literal, cast
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import Config
import state as research_state
import prompts
import utils


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
    state: research_state.ResearchInputState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    cfg = Config.from_runnable_config(config)

    llm = init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
    model = llm.with_structured_output(ClarifyUserRequestOutputSchema)

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

    result = cast(ClarifyUserRequestOutputSchema, result)

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
    state: research_state.ResearchInputState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    cfg = Config.from_runnable_config(config)

    llm = init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
    model = llm.with_structured_output(ResearchBriefOutputSchema)

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

    result = cast(ResearchBriefOutputSchema, result)

    return Command(
        update={"messages": [AIMessage(content=result.research_brief)]}, goto="__end__"
    )
