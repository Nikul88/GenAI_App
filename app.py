import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Prompt Template
prompt = PromptTemplate.from_template(
    "Tell me about {topic} in {no_of_words} words in {language} in a concise manner."
)

# LLM initialization
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4",
    temperature=0.7,
)

# LLMChain
final_llm = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("Azure GPT-4 Topic Explainer")
st.write("Provide a topic, desired word count, and language for a concise explanation.")

with st.form("input_form"):
    topic = st.text_input("Enter a topic:")
    no_of_words = st.number_input("Number of words:", min_value=10, max_value=1000, value=100)
    language = st.text_input("Language (e.g., English, Hindi, Spanish):")
    submitted = st.form_submit_button("Generate")

if submitted:
    if topic.strip() == "" or language.strip() == "":
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Generating response..."):
            result = final_llm.invoke({
                "topic": topic,
                "no_of_words": str(no_of_words),
                "language": language
            })
        st.subheader("GPT Response:")
        st.write(result["text"])