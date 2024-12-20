# FOREdiction

**FOREdiction** is an advanced web application designed to provide real-time data analysis, predictions, and visualizations. Built using **FastAPI** on the backend and **React** for the frontend, it integrates seamlessly with **OpenAI’s GPT models** and offers powerful data analysis and forecasting capabilities. 

The app is structured to support multiple data endpoints, including file uploads, real-time WebSocket communication for chat, data visualization, and forecasting. It allows users to upload files, initiate real-time chat sessions, and interact with various data visualizations and predictive models.

### Project Highlights:

- **Backend**: The backend is built using **FastAPI**, a high-performance web framework for building APIs with Python. It integrates with **OpenAI’s GPT models** for conversational AI capabilities and predictive analytics. Although PostgreSQL is not currently involved in the app's runtime, placeholders are configured to allow seamless database integration in future versions.
  
- **Frontend**: The frontend is developed with **React** and bootstrapped using **Create React App**, providing a responsive and dynamic user interface. It communicates with the backend to facilitate real-time data interactions and visualizations.

- **PostgreSQL Placeholder**: While PostgreSQL is not currently utilized in the application, database configurations are set up in anticipation of future data storage needs. This setup ensures that the app is easily extendable for storing and querying structured data.

### Features:

1. **Real-time Data Communication**: 
   - **WebSocket connections** are used for real-time chat and interactive data visualizations. 
   - The backend exposes endpoints for various graphing and data analysis tasks, such as time-series analysis, model selection, and outlier detection.

2. **File Upload**: 
   - Users can upload files to the backend, which can then be ingested for processing, analysis, or prediction.

3. **OpenAI Integration**:
   - Seamless integration with OpenAI’s **GPT-4o and O1** models for advanced data processing, predictions, and conversational AI capabilities.

4. **API Endpoints**:
   - The backend provides several WebSocket and HTTP endpoints, including file ingestion, chat communication, and various graphing and forecasting models. The frontend communicates with these endpoints for live data updates.

### Project Structure:

- **Backend**:
  - The backend utilizes FastAPI and Docker, exposing API endpoints for file ingestion, chat functionality, and various data visualizations. 
  - PostgreSQL integration is planned for future revisions, with configuration placeholders already set up.

- **Frontend**:
  - The frontend is a React application built using **Create React App**. It provides the user interface to interact with the backend services, including real-time WebSocket connections and API requests for data processing.

- **Docker Support**:
  - Dockerfiles are provided for both the frontend and backend to facilitate easy deployment and testing in containerized environments.

### Running the Application

1. **Build and start the containers**:

   From the root of your **FOREdiction** project, run the following command to build and start all services (frontend, backend, and PostgreSQL):

   ```bash
   docker-compose up --build
   ```

   - The `--build` flag ensures that Docker rebuilds the images from the Dockerfiles in the `frontend` and `backend` directories.
   - This command will start the backend on port `8000`, the frontend on port `3000`, and PostgreSQL in the background.

2. **Access the Application**:

   - **Frontend**: Open your browser and navigate to [http://localhost:3000](http://localhost:3000). This will load the **React** frontend.
   - **Backend API**: The FastAPI backend will be available at [http://localhost:8000](http://localhost:8000).
     - You can also access the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

3. **Stop the Application**:

   To stop all services, press `Ctrl+C` in the terminal where Docker Compose is running, or run the following command:

   ```bash
   docker-compose down
   ```

   This will stop and remove all the containers, but the data in PostgreSQL will persist because of the volume configuration.
