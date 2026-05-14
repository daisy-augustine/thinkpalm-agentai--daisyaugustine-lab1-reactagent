%%writefile README.md
# Flight Tracker

This is an AI-powered live flight tracking application built with Streamlit, Groq, AviationStack, and ADS-B data. It allows users to inquire about flight details and track live aircraft positions using natural language.

## Features

*   **Flight Details Lookup:** Get schedule, airline, and status by IATA flight number.
*   **Live Position Tracking:** Track aircraft in real-time using ADS-B data sources.
*   **Airport Information:** Retrieve airport details by IATA code.
*   **AI Agent:** Uses a Groq-powered ReAct agent for conversational interaction and tool calling.
*   **Interactive Map:** Visualizes flight routes and live positions.

## Setup and Installation

### 1. API Keys

This application requires API keys for the following services:

*   **AviationStack API Key:** Obtain a free API key from [AviationStack](https://aviationstack.com/).
*   **Groq API Key:** Obtain a free API key from [console.groq.com/keys](https://console.groq.com/keys).

**Store these keys securely in Google Colab Secrets:**

1.  In your Colab notebook, click the "🔑" icon in the left sidebar to open the Secrets manager.
2.  Add a new secret named `AVIATIONSTACK_API_KEY` and paste your AviationStack key.
3.  Add another secret named `GROQ_API_KEY` and paste your Groq API key.

### 2. Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Running the Streamlit App

To run the Streamlit application in a local environment or a cloud service like Google Colab:

1.  Ensure you have `cloudflared` installed and in your PATH (for public access in Colab).
2.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

    In Google Colab, a dedicated cell is usually provided to launch the app, handling `cloudflared` for a public URL.

## Project Structure

*   `app.py`: The main Streamlit application code, including AI agent logic, tool definitions, and UI.
*   `requirements.txt`: Lists all Python dependencies.
*   `README.md`: This file, providing project overview and setup instructions.
