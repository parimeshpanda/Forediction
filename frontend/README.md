
---

# Getting Started with FOREdiction Frontend

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser. The page will reload when you make changes.

### `npm test`

Launches the test runner in interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
Your app is ready to be deployed! See [deployment instructions](https://facebook.github.io/create-react-app/docs/deployment) for more details.

### `npm run eject`

**Warning**: This is a one-way operation. Once you `eject`, you cannot go back!  
If you need full control over the configuration (webpack, Babel, ESLint, etc.), use `eject`. Otherwise, it’s recommended to stick with the default setup.

---

## API Endpoints Configuration

To interact with the backend, configure the following endpoints in your application. Replace the placeholders `<YOUR_BACKEND_URL>` with the actual base URL of your backend server.

```javascript
export const apiEndpoints = {
    uploadFile: '<YOUR_BACKEND_URL>/api/v1/ingestion/',  // File upload endpoint
    chatWebsocket: '<YOUR_BACKEND_URL>/api/v1/query/ws/chat',  // WebSocket for chat
    graph1: '<YOUR_BACKEND_URL>/api/v1/query/ws/timeseries',  // WebSocket for time-series data
    graph2: '<YOUR_BACKEND_URL>/api/v1/query/ws/train_test_data',  // WebSocket for train/test data
    graph3: '<YOUR_BACKEND_URL>/api/v1/query/ws/outliers',  // WebSocket for outlier detection
    graph4: '<YOUR_BACKEND_URL>/api/v1/query/ws/model_selection',  // WebSocket for model selection
    graph5: '<YOUR_BACKEND_URL>/api/v1/query/ws/forecasting_modified',  // WebSocket for forecasting
};
```

### Endpoint Descriptions

- **`uploadFile`**: POST endpoint for file uploads (e.g., data ingestion).
- **`chatWebsocket`**: WebSocket endpoint for real-time chat.
- **`graph1` - `graph5`**: WebSocket endpoints for various data visualizations and analysis (e.g., time-series, train/test data, outliers, model selection, forecasting).

### Instructions

1. **Replace the placeholders**: Change `<YOUR_BACKEND_URL>` in the `apiEndpoints` object to the base URL of your backend (e.g., `https://your-backend.com`).
2. **Example usage**: Once configured, you can use these endpoints in your frontend code to interact with the backend for tasks like file upload, real-time chat, or data visualization.

```javascript
// Example of connecting to the chat WebSocket
const socket = new WebSocket(apiEndpoints.chatWebsocket);
socket.onopen = () => {
    console.log("WebSocket connection established!");
    socket.send(JSON.stringify({ message: "Hello, world!" }));
};
```

---

## Learn More

- [Create React App Documentation](https://facebook.github.io/create-react-app/docs/getting-started)
- [React Documentation](https://reactjs.org/)

For more advanced configurations, deployment steps, and troubleshooting, refer to the official documentation.

---