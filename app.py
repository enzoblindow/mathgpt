import os

import streamlit as st

from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from dotenv import load_dotenv
load_dotenv()


# page setup
st.set_page_config(
    page_title="MathGPT", 
    page_icon="🤖", 
    initial_sidebar_state="collapsed"
)
st.title("🤖 MathGPT")

# api key management
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# LLM + tools setup
llm = OpenAI(
    temperature=0,  # TODO: move to a configurable slider, consider adding other passable arguments too
    openai_api_key=OPENAI_API_KEY, 
    streaming=True
)
search = DuckDuckGoSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

# Initialize agent, TODO: modify, try out ReAct instead
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)


if prompt := st.chat_input():
    st.chat_message("user", avatar="🫡").write(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
