# 1. Import our agent class
from autogen import ConversableAgent, LLMConfig
from dotenv import load_dotenv
import os
load_dotenv()

# 2. Define our LLM configuration for OpenAI's gpt-5 mini
#    uses the OPENAI_API_KEY environment variable
llm_config = llm_config = LLMConfig(config_list={"api_type": "google", "model": "gemini-2.0-flash-lite","api_key":os.getenv("GEMINI_API_KEY")})

# 3. Create our LLM agent
my_agent = ConversableAgent(
    name="helpful_agent",
    system_message="You are a poetic AI assistant, respond in rhyme.",
    llm_config=llm_config,
)

# 4. Run the agent with a prompt
response = my_agent.run(
    message="In one sentence, what's the big deal about AI?",
    max_turns=3,
    user_input=True
)

# 5. Iterate through the chat automatically with console output
response.process()

# 6. Print the chat
print(response.messages)