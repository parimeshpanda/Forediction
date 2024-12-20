import { Snackbar, Alert } from '@mui/material';
import { appConfigs } from '../constants/ApplicationConstants';


const PageMessage = ({show, message, type, clearPageMessage}) => {
    
    return (
        <Snackbar anchorOrigin={{ vertical: 'top', horizontal: 'left' }} open={show} autoHideDuration={appConfigs.errorMessageHideDuration} onClose={clearPageMessage}>
          <Alert
            severity={type}
            variant="filled"
          >
              {message}
          </Alert>
      </Snackbar>
    );
}

export default PageMessage;