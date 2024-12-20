import { Box, Typography, Backdrop, CircularProgress } from '@mui/material';
import Header from './Header';
import PageMessage from './PageMessage';
import { useDispatch, useSelector } from 'react-redux';
import { appConfigs } from '../constants/ApplicationConstants';
import { setPageMessage } from '../store/ApplicationStore';
import video from '../assets/images/background-video.mp4'

const PageLayout = ({ header, footer, children, nogutter }) => {
    const { processing, pageMessage } = useSelector(state=> state.app)

    const dispatch = useDispatch();

    function handlePageMessageClose(){
        dispatch(setPageMessage(null))
    }

    return(
        <>
            <Box style={!nogutter?{paddingLeft: '2rem', paddingRight: '2rem'}:null} sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', overflow: 'hidden'}}>
                <video autoPlay loop muted style={{ opacity: '0.8', position: 'absolute', top: '50%', left: '50%', width: '100%', height: '100%', transform: 'translate(-50%, -50%)', objectFit: 'cover', overflow: 'hidden', background:'#000000', zIndex: -1}}>
                    <source src={video} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
                {header&& <Header />}
                <Box  sx={{ marginTop: '4rem', flexGrow: 1, height:'100%', width: '100%', overflow: 'auto', background: 'rgba(0, 0, 0, 0.5)' }}>
                    {children}
                </Box>
                {footer&& <footer style={{background: 'rgba(0, 0, 0, 0.5)'}}>
                    <Typography color='white' sx={{px: '2rem',paddingBottom:'10px'}} variant="subtitle2" style={{float: 'right'}} fontSize={'10px'}>{appConfigs.footerText}</Typography>
                </footer>}
            </Box>
            <Backdrop sx={{ color: '#fff', zIndex: 2000}} open={processing}>
                <CircularProgress color="inherit" />
            </Backdrop>
            {pageMessage&&<PageMessage show={pageMessage?true:false} message={pageMessage.message} type={pageMessage.type} clearPageMessage={handlePageMessageClose}/>}
        </>
    )
}

export default PageLayout;