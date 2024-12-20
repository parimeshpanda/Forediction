# Database Configuration
POSTGRES_USERNAME = "your_postgres_username"  # Replace with your PostgreSQL username
POSTGRES_PASSWORD = "your_postgres_password"  # Replace with your PostgreSQL password
POSTGRES_HOST = "your_postgres_host"  # Replace with your PostgreSQL host (e.g., localhost or an IP address)
POSTGRES_PORT = "5432"  # Default PostgreSQL port (can be changed if necessary)
POSTGRES_DATABASE = (
    "your_postgres_database"  # Replace with your PostgreSQL database name
)

# OpenAI API Configuration
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your Azure OpenAI API key
AZURE_OPENAI_ENDPOINT = "your_azure_openai_endpoint"  # Replace with your Azure OpenAI endpoint (e.g., https://your-resource-name.openai.azure.com/)
OPENAI_API_VERSION = "your_openai_api_version"  # The version of the OpenAI API you're using, e.g., "2023-03-15-preview"
OPENAI_MODEL_NAME = (
    "gpt-4"  # Replace with the model you want to use (e.g., gpt-4, gpt-3.5-turbo)
)
OPENAI_API_TYPE = "azure"  # API type for Azure, this could be 'azure' or 'openai' depending on your setup

# Additional OpenAI Model Configuration
OPENAI_MODEL_NAME_O1 = (
    "gpt-4"  # Another model you might use in a different context or environment
)
