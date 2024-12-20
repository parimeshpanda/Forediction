import { AppBar, Toolbar, IconButton, Typography, Menu, MenuItem, Box } from '@mui/material';
import { AccountCircle, Person } from '@mui/icons-material';
import { useState } from 'react';
import logo from '../assets/images/genpactlogo.svg';
import { GradientIcon } from './GradientIcon';

const Header = () => {
    const [anchorEl, setAnchorEl] = useState(null);
 
    function handleMenuOpen(event){
        setAnchorEl(event.currentTarget);
    }

    function handleMenuClose(){
        setAnchorEl(null);
    }


    return(
        <AppBar position="fixed" sx={{px: '2rem', height: '4rem', backgroundColor: 'rgba(0, 0, 0, 0.8)'}}>
            <Toolbar disableGutters>
                <Typography sx={{ flexGrow: 1, cursor: 'pointer' }}>
                    <Box display='flex' flexDirection='row'>
                        <img alt="Logo" src={logo} style={{
                            width: 120,
                            height: 30,
                            marginTop: 2
                        }} />
                        <Typography variant="h6" style={{color: 'grey', marginLeft: '0.1rem'}}>|</Typography>
                        <Typography variant="h6" style={{color: '#ff545f', marginLeft: '0.5rem'}}>FORE</Typography>
                        <Typography variant="h6" style={{color: '#00aecf'}}>diction</Typography>
                    </Box>
                </Typography>
                <Box>
                    <IconButton size="large" aria-label="account of current user"  onClick={handleMenuOpen} color="inherit">
                        <AccountCircle />
                    </IconButton>
                    <Menu id="menu-appbar" anchorEl={anchorEl} keepMounted open={Boolean(anchorEl)} onClose={handleMenuClose}>
                        <MenuItem><GradientIcon icon={Person} /><Typography sx={{ml:3}}>User</Typography></MenuItem>
                    </Menu>
                </Box>
            </Toolbar>
        </AppBar>
    )
}

export default Header;