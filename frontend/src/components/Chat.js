import { FileUploadOutlined, Send, StopCircleOutlined } from "@mui/icons-material"
import { Box, IconButton, InputAdornment, styled, TextField, Tooltip, Typography } from "@mui/material"
import { useEffect, useState } from "react";
import { GradientIcon } from "./GradientIcon";
import { apiEndpoints } from "../constants/ApplicationConstants";
import chatlogo from "../assets/images/genpact.png";
import { useDispatch } from "react-redux";
import { setPageMessage, setProcessing } from "../store/ApplicationStore";
import axios from 'axios';
import { Remarkable } from 'remarkable';

const VisuallyHiddenInput = styled('input')({
    clip: 'rect(0 0 0 0)',
    clipPath: 'inset(50%)',
    height: 1,
    overflow: 'hidden',
    position: 'absolute',
    bottom: 0,
    left: 0,
    whiteSpace: 'nowrap',
    width: 1,
});

export const ChatComponent = ({onTerminate}) => {
    const [question, setQuestion] = useState('');
    const [chatHistory, setChatHistory] = useState([{question: 'Please upload your timeseries input file.', response: 'Waiting for Document Upload...'}]); 
    const [connection, setConnection] = useState(null);
    const [file, setFile] = useState(null);
    const [enableInterruption, setEnableInterruption] = useState(false);

    const dispatch = useDispatch();
    const remark= new Remarkable();

    useEffect(() => {
        const ws = new WebSocket(apiEndpoints.chatWebsocket);
        ws.onopen = () => setConnection(ws);
        ws.onmessage = (event) => {
            const data = event.data; 
            setChatHistory((prevChat) => {
                const lastItemIndex = prevChat.length - 1;
                if (lastItemIndex >= 1) {
                    const updatedChat = [...prevChat];
                    const currentResponse = updatedChat[lastItemIndex]?.response;
                    if(data !== 'RESPONSE_GENERATION_COMPLETED'){
                        if (!currentResponse?.endsWith(data)) {
                            updatedChat[lastItemIndex].response += data;
                        }
                    }else{
                        setEnableInterruption(false);
                    }
                    return updatedChat;
                } else {
                    return prevChat;
                }
            });
        };
    
        ws.onclose = () => {
            setEnableInterruption(false);
            console.log("WebSocket connection closed");
        };
    
        ws.onerror = (error) => {
            setEnableInterruption(false);
            console.error("WebSocket error:", error.type)
        };
    
        return () => ws.close();
    }, []);
    

    const handlePostQuestion = () => {
        if (connection && question.trim()){
            setEnableInterruption(true);
            const newQuestion = { id : chatHistory.length + 1, question: question, response: ""};
            setChatHistory((prevChat) => [...prevChat, newQuestion]);
            const temp= `The following is the path to CSV: src/original_df/${file.name} ` + question;
            connection.send(JSON.stringify({ temp }));
            setQuestion("");
        }
    }
    const handleFile = async (file) => {
        if(file){
           if(file?.name?.split('.')[1] === 'csv'){
                try{
                    dispatch(setProcessing(true));
                    const formData = new FormData();
                    formData.append('file', file);
                    const response = await axios.post(apiEndpoints.uploadFile, formData);
                    if((response.status === 200 || response.status === 201) && response?.data?.status === 'success'){
                        setFile(file);
                        let temp= chatHistory;
                        temp[0].response = file.name + ' uploaded successfully. Please ask your question.';  
                        setChatHistory(temp);
                        dispatch(setPageMessage({type: 'success', message: 'File uploaded successfully.'}));
                        dispatch(setProcessing(false));
                    }else{
                        setFile(null);
                        dispatch(setPageMessage({type: 'error', message: 'Failed to upload file.'}));
                        dispatch(setProcessing(false));
                    }
                }catch(error){
                    setFile(null);
                    dispatch(setPageMessage({type: 'error', message: 'Failed to upload file.'}));
                    dispatch(setProcessing(false));
                }
            }else{
                dispatch(setPageMessage({type: 'error', message: 'Please upload a csv file.'}));
            }
        }else{
            setFile(null);
            dispatch(setPageMessage({type: 'error', message: 'Please upload a file.'}));
        }
    }

    return (
        <Box p={2} display='flex' flexDirection='column'>
            <Box py={0.5} textAlign='center' sx={{background: 'linear-gradient(90deg, #FE1F4B 0%, #FE742A 100%)'}}>
                <Typography variant='body1' fontWeight='bold'>Iterative Planning - Reasoning</Typography>
            </Box>
            <Box sx={{py:1, height: 'calc(100vh - 14rem)', overflowY: 'scroll'}}>
                {chatHistory.map((item,index) => (
                    <Box key={item.id}>
                        <Box display='flex' flexDirection='row'>
                            {index!==0?<Typography height='38px' width='35px' sx={{px:1.5, py:1, background: '#FE1F4B', borderRadius: '3px'}} color='white' variant='body2'>U</Typography>:<img height='40px' width='35px' alt="logo" src={chatlogo} style={{marginTop: 1}} />}
                            <Typography mt={1} ml={2} color='white' variant='body2'>{item.question}</Typography>
                        </Box>
                        <Box mt={1} display='flex' flexDirection='row'>
                            <img height='40px' width='35px' alt="logo" src={chatlogo} style={{marginTop: 1}} />
                            {!item.response && <Typography mt={1} ml={2} color='white' variant='body2'>Waiting for response...</Typography>}
                            {item.response &&  <div style={{marginLeft: '1rem', fontSize: 14}} class="markupTable" dangerouslySetInnerHTML={{ __html: remark.render(item.response.replace(/(###)/g, `\n###`))}} ></div>}
                        </Box>
                    </Box>
                ))}
            </Box>
            <Box sx={{mt:1, background: 'rgba(0, 0, 0, 0.8)'}}>
                <TextField multiline rows={2} InputLabelProps={{shrink: question?true:false, style: {color: '#888', fontSize: '14px'}}} disabled={enableInterruption} value={question} onKeyUp={(e)=>e.keyCode === 13&&handlePostQuestion()} onChange={(e)=>setQuestion(e.target.value)} size='small' label={!file?"Upload the file":"Please type your plan here"} variant="outlined" fullWidth
                    InputProps={{
                        endAdornment: (
                        <InputAdornment position="end">
                            {!file&&<Tooltip title="Upload"><IconButton onClick={() => document.getElementById('fileInput').click()}>
                                <GradientIcon icon={FileUploadOutlined} />
                                <VisuallyHiddenInput type="file" id="fileInput" onChange={(e)=>handleFile(e.target.files[0])} />
                            </IconButton></Tooltip>}
                            {file&&<Tooltip title="Send"><IconButton onClick={()=>handlePostQuestion()}>
                                <GradientIcon icon={Send} />
                            </IconButton></Tooltip>}
                            {enableInterruption&&<Tooltip title="Terminate Response"><IconButton onClick={()=>{
                                    //setFile(null); 
                                    setEnableInterruption(false)
                                    //setChatHistory([{question: 'Please upload your timeseries input file.', response: null}]);
                                    connection.send(JSON.stringify('END'));
                                    //onTerminate();
                                }}>
                                <GradientIcon icon={StopCircleOutlined} />
                            </IconButton></Tooltip>}
                        </InputAdornment>
                        )
                    }}
                />
            </Box>
        </Box>
    )
}