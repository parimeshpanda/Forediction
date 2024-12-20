import { configureStore } from "@reduxjs/toolkit";
import applicationReducer from "./ApplicationStore";

const store = configureStore({
  reducer: {
    app: applicationReducer,
  }
});

export default store;