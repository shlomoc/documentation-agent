# Import necessary libraries and modules
from typing import Set
from consts import APP_HEADER
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

# Set up the Streamlit app header
st.header(APP_HEADER)

# Create an input field for the user's prompt
prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

# Initialize session state variables if they don't exist
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


# Function to create a formatted string of source URLs
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        source = source.replace('\\', '/')
        sources_string += f"{i + 1}. [{source}]({source})\n"

    return sources_string


# Main logic for handling user input and generating responses
if prompt:
    with st.spinner("Generating response.."):
        # Run the language model to generate a response
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        
        # Extract source URLs from the response
        sources = set(
            [doc.metadata["sourceURL"] for doc in generated_response["source_documents"]]
        )

        # Format the response with the generated text and sources
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        # Display the formatted response with clickable links
        st.markdown(formatted_response, unsafe_allow_html=True)

        # Update session state with the new interaction
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


# Display the chat history
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)
