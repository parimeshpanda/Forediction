Here’s the updated **README** for your **FOREdiction** project with the relevant API endpoints included:

---

# FOREdiction

**FOREdiction** is a FastAPI-based web application that integrates with OpenAI and PostgreSQL for real-time data analysis and prediction tasks.

## Getting Started

To set up and run **FOREdiction** locally, follow these steps:

### Prerequisites

1. **Python 3.8+** (recommended)
2. **PostgreSQL** database setup (locally or in the cloud)
3. **Azure OpenAI API** credentials
4. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Update the `config.py` file to include your database and OpenAI API configuration. Below are the placeholders you need to replace with your actual credentials:

```python
# Database Configuration
POSTGRES_USERNAME = "your_postgres_username"  # Replace with your PostgreSQL username
POSTGRES_PASSWORD = "your_postgres_password"  # Replace with your PostgreSQL password
POSTGRES_HOST = "your_postgres_host"  # Replace with your PostgreSQL host (e.g., localhost or an IP address)
POSTGRES_PORT = "5432"  # Default PostgreSQL port (can be changed if necessary)
POSTGRES_DATABASE = "your_postgres_database"  # Replace with your PostgreSQL database name

# OpenAI API Configuration
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your Azure OpenAI API key
AZURE_OPENAI_ENDPOINT = "your_azure_openai_endpoint"  # Replace with your Azure OpenAI endpoint (e.g., https://your-resource-name.openai.azure.com/)
OPENAI_API_VERSION = "your_openai_api_version"  # The version of the OpenAI API you're using, e.g., "2023-03-15-preview"
OPENAI_MODEL_NAME = "gpt-4"  # Replace with the model you want to use (e.g., gpt-4, gpt-3.5-turbo)
OPENAI_API_TYPE = "azure"  # API type for Azure, this could be 'azure' or 'openai' depending on your setup
```

### API Endpoints

Below are the primary API endpoints for **FOREdiction**. The FOREdiction API exposes several endpoints for various actions like interacting with models and retrieving graphs.

- uploadFile: **/api/v1/ingestion/** (POST)

- chatWebsocket: **/api/v1/query/ws/chat** (WebSocket)

- graph1: **/api/v1/query/ws/timeseries** (WebSocket)

- graph2: **/api/v1/query/ws/train_test_data** (WebSocket)

- graph3: **/api/v1/query/ws/outliers** (WebSocket)

- graph4: **/api/v1/query/ws/model_selection** (WebSocket)

- graph5: **/api/v1/query/ws/forecasting_modified** (WebSocket)

### Endpoint Descriptions

- **`uploadFile`**: POST endpoint for uploading data files to the backend (e.g., data ingestion).
- **`chatWebsocket`**: WebSocket for establishing a real-time chat connection with the backend.
- **`graph1`**: WebSocket for streaming time-series data visualizations.
- **`graph2`**: WebSocket for streaming train/test data for machine learning models.
- **`graph3`**: WebSocket for streaming outlier detection data.
- **`graph4`**: WebSocket for streaming model selection data.
- **`graph5`**: WebSocket for forecasting data streams.

### Running the Application

1. **Start the FastAPI server**:
   After you’ve updated the configuration, run the FastAPI application:

   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API Docs**:
   Once the server is running, open your browser and navigate to:
   [http://localhost:8000/docs](http://localhost:8000/docs)  
   This will display the Swagger UI with automatic documentation for all available endpoints.

### Docker Setup (Optional)

You can also run **FOREdiction** in a Docker container for easier deployment.

1. **Build the Docker image**:
   ```bash
   docker build -t forediction .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 8000:8000 forediction
   ```

Your application will now be accessible at [http://localhost:8000](http://localhost:8000).
