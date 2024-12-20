import { createSlice } from "@reduxjs/toolkit";
import { appConfigs } from "../constants/ApplicationConstants";

const initialState= {
  processing: false,
  pageMessage: null,
  records: null,
  pagination: appConfigs.tableAttributes
}

const appSlice = createSlice({
  name: "app",
  initialState: initialState,
  extraReducers: (builder) => {
    builder
     .addCase('LOGOUT', (state) => {
        Object.assign(state, initialState);
     })
  },
  reducers: {
    setProcessing: (state, action) => {
      state.processing = action.payload
    },
    setPageMessage: (state, action) => {
      state.pageMessage = action.payload
    },
    clearPageMessage:(state)=>{
      state.pageMessage = null
    },
    setRecords: (state, action) => {
      state.records = action.payload
    },
    setPagination: (state, action) => { 
      state.pagination = action.payload
    },
    clearRecords: (state) => {
      state.records = null;
    },
    clearPagination: (state) => { 
      state.pagination = appConfigs.tableAttributes;
    }
  },
});

export const { setProcessing, setPageMessage, clearPageMessage, setRecords, setPagination, clearPagination, clearRecords } = appSlice.actions;
export default appSlice.reducer;