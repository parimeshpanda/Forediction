import Plot from "react-plotly.js"
import { sampleGraph } from "../constants/ApplicationConstants"
import { Box } from "@mui/material"

export const Graph8 = () => {
    return (
        <Box>
            {graphdata && <Plot data={graphdata?.data} layout={graphdata?.layout} />}
        </Box>
    )
}