import React from 'react';
import { SvgIcon } from '@mui/material';
import { defaultAppTheme } from '../constants/ApplicationConstants';

export const GradientIcon = ({ icon: IconComponent, props }) => {
  return (
    <SvgIcon {...props}>
      <defs>
        <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: defaultAppTheme.palette.primary.main, stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: defaultAppTheme.palette.secondary.main, stopOpacity: 1 }} />
        </linearGradient>
      </defs>
      {React.cloneElement(<IconComponent />, { style: { fill: 'url(#gradient1)' } })}
    </SvgIcon>
  );
};
