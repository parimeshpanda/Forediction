from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, AIMessagePromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent
import config as cfg


class AgentTemplate:

    llm = AzureChatOpenAI(
    api_key=cfg.OPENAI_API_KEY,
    azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
    api_version=cfg.OPENAI_API_VERSION,
    model=cfg.OPENAI_MODEL_NAME,
    openai_api_type=cfg.OPENAI_API_TYPE,
    temperature=0.1,
    streaming=True,
    )

    llm_o1 = AzureChatOpenAI(
        api_key=cfg.OPENAI_API_KEY,
        azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
        api_version=cfg.OPENAI_API_VERSION,
        model=cfg.OPENAI_MODEL_NAME_O1,
        openai_api_type=cfg.OPENAI_API_TYPE,
        streaming=True
        # temperature=0.2,
    )

    def __init__(
            self,
            system_prompt: str,
            toolkit: list[StructuredTool] | None = None,
            output_parser: BaseModel | None = None
        ) -> None:
        self.system_prompt = system_prompt
        self.toolkit = toolkit
        self.output_parser = output_parser

    def create_agent_with_tools(self):
        prompt = ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=self.system_prompt + """
                        ONLY respond to the part of query relevant to your purpose.
                        IGNORE tasks you can't complete.
                        Use the following context to answer your query """)),
                AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=['chat_history'], template='chat_history: {chat_history}')),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['user_input'], template='{user_input}')),
                MessagesPlaceholder(variable_name='agent_scratchpad')]
        )
        agent = create_tool_calling_agent(AgentTemplate.llm, self.toolkit, prompt)
        executor = AgentExecutor(agent=agent, tools=self.toolkit,
                return_intermediate_steps= True, verbose = True)
        return executor

    def create_agent_without_tools(self, agent_description: str, verbose: bool = True, ):
        executor = initialize_agent(
            agent=""

        )