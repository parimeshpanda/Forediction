import Plot from "react-plotly.js"
import { sampleGraph } from "../constants/ApplicationConstants"
import { Box } from "@mui/material"

export const Graph9 = () => {
    return (
        <Box>
            {graphdata && <Plot data={graphdata?.data} layout={graphdata?.layout} />}
        </Box>
    )
}