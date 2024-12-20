import Plot from "react-plotly.js"
import { apiEndpoints } from "../constants/ApplicationConstants"
import { Box } from "@mui/material"
import { useCallback, useEffect, useState } from "react";

export const Graph6 = ({clearData, onDataLoad}) => {
    const [ graphdata, setGraphdata] = useState(null);
    const [ connection, setConnection] = useState(null);

    const getGraphData = useCallback(()=>{
        if (connection){
            connection.send(null);
        }
    },[connection]);
    useEffect(() => {
        const ws = new WebSocket(apiEndpoints.forcastingFsocket);
        ws.onopen = () => setConnection(ws);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data); 
            setGraphdata(data);
            data&&onDataLoad({id: 'G6', data: data});
        };
        ws.onclose = () =>  console.log("WebSocket connection closed");    
        ws.onerror = (error) =>  console.error("WebSocket error:", error.type);    
        return () => ws.close();  
    }, [onDataLoad]);

    useEffect(()=>{
        getGraphData();
    },[getGraphData])

    useEffect(()=>{
        if(clearData){
            setGraphdata(null);
        }
    },[clearData])

    return (
        <Box>
            {graphdata && <Plot data={graphdata?.data} layout={{...graphdata?.layout, width: 900, height: 600}} />}
        </Box>
    )
}