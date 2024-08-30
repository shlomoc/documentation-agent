from typing import Set

import streamlit as st

from consts import APP_HEADER
from backend.core import run_llm


def create_sources_string(source_urls: Set[str]) -> str:
    """
    Create a formatted string of source URLs.

    Args:
    source_urls (Set[str]): A set of source URLs.

    Returns:
    str: A formatted string of numbered source URLs.
    """
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        source = source.replace("\\", "/")
        sources_string += f"{i + 1}. [{source}]({source})\n"
    return sources_string


def chat():
    """
    Main function to run the chat interface.
    """
    # Set the header of the Streamlit app
    st.header(APP_HEADER)

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Enter your prompt here.."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response.."):
                # Run the language model to generate a response
                generated_response = run_llm(
                    query=prompt,
                    chat_history=[
                        (msg["role"], msg["content"])
                        for msg in st.session_state.chat_history
                        if msg["role"] in ["human", "ai"]
                    ],
                )

                # Extract source URLs from the response
                sources = set(
                    [
                        doc.metadata["sourceURL"]
                        for doc in generated_response["source_documents"]
                    ]
                )

                # Format the response with the generated text and sources
                formatted_response = f"{generated_response['result']}\n\n{create_sources_string(sources)}"

                # Display the formatted response with clickable links
                st.markdown(formatted_response, unsafe_allow_html=True)

        # Add AI response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": formatted_response}
        )


if __name__ == "__main__":
    chat()
