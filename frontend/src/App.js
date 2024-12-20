import { Routes, Route } from 'react-router-dom';
import { HomeScreen } from './screens/HomeScreen';
import './App.css';
import { ThemeProvider } from '@emotion/react';
import { createTheme, CssBaseline } from '@mui/material';
import { defaultAppTheme } from './constants/ApplicationConstants';

function App() {
  return (
    <ThemeProvider theme={createTheme(defaultAppTheme)}>
      <CssBaseline />
      <Routes>
        <Route path="/" element={<HomeScreen />} />
      </Routes>
    </ThemeProvider>
  );
}

export default App;
