import PageLayout from '../components/PageLayout';
import Grid2 from '@mui/material/Unstable_Grid2';
import { ChatComponent } from '../components/Chat';
import { Graph1 } from '../components/Graph1';
import { Graph2 } from '../components/Graph2';
import { Graph3 } from '../components/Graph3';
import { Graph4 } from '../components/Graph4';
import { Graph5 } from '../components/Graph5';
import { Box, Button, Tab, Tabs, Tooltip } from '@mui/material';
import { Graph6 } from '../components/Graph6';
import { Graph7 } from '../components/Graph7';
import { useState } from 'react';
import { ArrowBack, ArrowForward } from '@mui/icons-material';
import { GradientIcon } from '../components/GradientIcon';

export const HomeScreen = () => {
   const [chatTerminated, setChatTerminated] = useState(false);
   const [tabs, setTabs] = useState([]);
   const [activeTab, setActiveTab] = useState(0);

   const updateTabs = (graph) => {
      const { id, data } = graph;
      const existingIndex = tabs.findIndex((tab) => tab.id === id);
      let updatedTabs;
      if (existingIndex !== -1) {
          updatedTabs = [...tabs];
          updatedTabs[existingIndex] = { id, data };
      } else {
          updatedTabs = [...tabs, { id, data }];
      }
      updatedTabs.sort((a, b) => {
          const idA = parseInt(a?.id?.replace('G', ''), 10);
          const idB = parseInt(b?.id?.replace('G', ''), 10);
          return idA - idB;
      });
      setTabs(updatedTabs);
      console.log(updatedTabs);
  };
  

   const handleNext = () => {
      setActiveTab((prevIndex) => (prevIndex + 1) % tabs.length);
    };
  
    const handlePrevious = () => {
      setActiveTab(
        (prevIndex) => (prevIndex - 1 + tabs.length) % tabs.length
      );
    };

   return (
        <PageLayout header nogutter footer>
            <Grid2 container>
                <Grid2 item md={4} sx={{border: '1px solid grey', borderRadius: '8px', height: 'calc(100vh - 5rem)'}}>
                    <ChatComponent onTerminate={()=>{setChatTerminated(true); setActiveTab(0); setTabs([])}} />
                </Grid2>
                <Grid2 item md={7.75} sx={{p:1, mx: 0.25, border: '1px solid grey', borderRadius: '8px', width: '100%', height: 'calc(100vh - 5rem)', overflowY: 'auto'}}>
                     <Tabs value={activeTab} onChange={(_, newIndex) => setActiveTab(newIndex)} variant="scrollable" scrollButtons="auto"allowScrollButtonsMobile>
                        {tabs.map((item, index) => {
                        const isDisabled = tabs[index] === undefined;
                           return (
                              <Tooltip key={item?.id} title={item?.data?.layout?.title?.text}><Tab
                                 label={ <Box sx={{ width: 120, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{item?.data?.layout?.title?.text}</Box>}
                                 disabled={isDisabled}
                                 sx={{
                                    '&:not(:last-child)': {
                                       borderRight: '1px solid grey',
                                    },
                                    '&.Mui-disabled': {
                                       color: 'grey',
                                    },
                                 }}
                              /></Tooltip>
                           );
                        })}
                     </Tabs>
                     <Box display= 'flex' alignItems= 'center' justifyContent= 'center' width= '100%' mt={4}>
                        {tabs.length>0 && <Button disabled={!tabs.length>1} onClick={handlePrevious} sx={{ minWidth: 'auto'}}><GradientIcon icon={ArrowBack} /></Button>}
                        <Box display= 'flex' justifyContent= 'center' alignItems= 'center' width= '90%'>
                           <Box display={activeTab===0? 'block' : 'none'}><Graph1 clearData={chatTerminated} onDataLoad={updateTabs} /></Box>
                           <Box display={activeTab===1? 'block' : 'none'}><Graph2 clearData={chatTerminated} onDataLoad={updateTabs} /></Box>
                           <Box display={activeTab===2? 'block' : 'none'}><Graph3 clearData={chatTerminated} onDataLoad={updateTabs} /></Box>
                           <Box display={activeTab===3? 'block' : 'none'}><Graph4 clearData={chatTerminated} onDataLoad={updateTabs} /></Box>
                           <Box display={activeTab===4? 'block' : 'none'}><Graph5 clearData={chatTerminated} onDataLoad={updateTabs} /></Box>
                        </Box>
                        {tabs.length>0 && <Button disabled={!tabs.length>1} onClick={handleNext} sx={{ minWidth: 'auto'}}><GradientIcon icon={ArrowForward} /></Button>}
                     </Box>
                </Grid2>
            </Grid2>
        </PageLayout>
   )
}