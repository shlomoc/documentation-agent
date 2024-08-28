# Documentation Chatbot

This project is a documentation chatbot that uses Langchain to process and retrieve information from Salesforce Einstein GenAI documentation.

## Features

- Uses Langchain for document processing and retrieval
- Integrates with OpenAI for embeddings and question answering
- Stores document embeddings in Pinecone for efficient retrieval
- Provides a Streamlit-based user interface for interacting with the chatbot
- Supports monitoring and debugging with Langsmith

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/shlomoc/documentation-chatbot.git
   ```

2. Navigate to the project directory:
   ```bash
   cd documentation-chatbot
   ```

3. Create a `.env` file in the project root directory and add your API keys:
   ```bash
   touch .env
   ```
   Open the `.env` file and add the following lines, replacing `your_api_key_here` with your actual API keys:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   PINECONE_API_KEY=your_api_key_here
   FIRECRAWL_API_KEY=your_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=your_project_name_here
   ```

4. Install Pipenv if you haven't already:
   ```bash
   pip install pipenv
   ```

5. Install the required dependencies using Pipenv:
   ```bash
   pipenv install
   ```

6. Activate the Pipenv shell:
   ```bash
   pipenv shell
   ```

7. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

8. Open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501).

9. Type your question in the space provided to get a response.

## Monitoring with Langsmith

This project uses Langsmith for monitoring and debugging Langchain applications. To use Langsmith:

1. Sign up for a Langsmith account at https://smith.langchain.com/
2. Obtain your Langsmith API key
3. Add your Langsmith API key to the `.env` file as shown in step 3 above
4. Set `LANGCHAIN_TRACING_V2=true` in your `.env` file to enable tracing
5. Specify your Langsmith project name using `LANGCHAIN_PROJECT` in the `.env` file

With these settings, your app's interactions will be logged to Langsmith, allowing you to monitor performance, debug issues, and optimize your Langchain application.


