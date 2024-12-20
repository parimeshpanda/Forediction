export const appRoutes= {
    home: "/",
}

export const apiEndpoints = {
    // URL for file upload API
    uploadFile: '<YOUR_BACKEND_URL>/api/v1/ingestion/',

    // WebSocket endpoint for chat functionality
    chatWebsocket: '<YOUR_BACKEND_URL>/api/v1/query/ws/chat',

    // WebSocket endpoint for time-series data
    graph1: '<YOUR_BACKEND_URL>/api/v1/query/ws/timeseries',

    // WebSocket endpoint for train/test data
    graph2: '<YOUR_BACKEND_URL>/api/v1/query/ws/train_test_data',

    // WebSocket endpoint for outlier detection
    graph3: '<YOUR_BACKEND_URL>/api/v1/query/ws/outliers',

    // WebSocket endpoint for model selection data
    graph4: '<YOUR_BACKEND_URL>/api/v1/query/ws/model_selection',

    // WebSocket endpoint for forecasting data
    graph5: '<YOUR_BACKEND_URL>/api/v1/query/ws/forecasting_modified',
}

export const appConfigs={
    footerText: "© 2024 Copyright Genpact. All Rights Reserved | Powered by Gen AI",
    errorMessageHideDuration: 4000
}

export const defaultAppTheme = {
    typography: {
        fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
        fontSize: 14,
    },
    palette: {
        primary: {
            main: "#FE1F4B",
            contrastText: "#FFFFFF",
        },
        secondary: {
            main: "#FE742A",
            contrastText: "#FFFFFF",
        },
        teritiary: {
            main: "#FF555F",
            contrastText: "#FFFFFF",
        },
        success: {
            main: "#1BA641",
            contrastText: "#FFFFFF",
        },
        error: {
            main: "#B43B44",
            contrastText: "#FFFFFF",
        },
        warning: {
            main: "#F2C43B",
            contrastText: "#FFFFFF",
        },
        background: {
            default: "#000000",
            paper: "#000000",
        },
        text: {
            primary: "#FFFFFF",
            secondary: '#FFFFFF'
        }
    }
 }

 export const gradientText= {
    background: `linear-gradient(90deg, ${defaultAppTheme.palette.primary.main} 0%, ${defaultAppTheme.palette.secondary.main} 100%)`,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
 }

 export const sampleGraph= [
        {
            x: [1, 2, 3],
            y: [2, 6, 3],
            type: 'scatter',
            mode: 'lines+markers',
            marker: {color: 'red'}
        },
        {
            type: 'bar', 
            x: [1, 2, 3], 
            y: [2, 5, 3]
        }
    ]
