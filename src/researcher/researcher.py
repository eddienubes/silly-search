import typing
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import state
from ..config import Config
from . import tools
from .. import tools as common_tools
from .. import prompts
from .. import utils


async def research(
    state: state.ResearcherState, config: RunnableConfig
) -> Command[typing.Literal["handle_researcher_tools"]]:
    cfg = Config.from_runnable_config(config)

    llm = (
        init_chat_model(model=cfg.xai_model_name, api_key=cfg.xai_api_key)
        .bind_tools([tools.search, common_tools.think])
        .with_retry(stop_after_attempt=cfg.max_llm_retries)
    )

    system_prompt = prompts.researcher_system_prompt.format(
        data=utils.get_readable_date(),
        mcp_prompt="",
    )
    messages = [SystemMessage(content=system_prompt)] + state.get(
        "researcher_messages", []
    )

    response = await llm.ainvoke(messages)

    return Command(
        goto="handle_researcher_tools", update={"researcher_messsages": [response]}
    )
