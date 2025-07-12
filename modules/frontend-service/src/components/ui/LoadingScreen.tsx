import React from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  LinearProgress,
  useTheme,
  alpha,
} from '@mui/material';
import { motion } from 'framer-motion';

interface LoadingScreenProps {
  message?: string;
  progress?: number;
  variant?: 'circular' | 'linear' | 'minimal';
  size?: 'small' | 'medium' | 'large';
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Loading...',
  progress,
  variant = 'circular',
  size = 'medium',
}) => {
  const theme = useTheme();

  const sizeMap = {
    small: 32,
    medium: 48,
    large: 64,
  };

  const containerVariants = {
    initial: { opacity: 0 },
    animate: { 
      opacity: 1,
      transition: { duration: 0.3 }
    },
    exit: { 
      opacity: 0,
      transition: { duration: 0.2 }
    }
  };

  const contentVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: { 
        delay: 0.1,
        duration: 0.4,
        ease: "easeOut"
      }
    }
  };

  if (variant === 'minimal') {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing(2),
        }}
      >
        <CircularProgress size={sizeMap[size]} />
      </Box>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: alpha(theme.palette.background.default, 0.9),
          backdropFilter: 'blur(4px)',
          zIndex: theme.zIndex.modal + 1,
        }}
      >
        <motion.div variants={contentVariants}>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 3,
              padding: theme.spacing(4),
              borderRadius: theme.shape.borderRadius * 2,
              backgroundColor: theme.palette.background.paper,
              boxShadow: theme.shadows[8],
              border: `1px solid ${theme.palette.divider}`,
              minWidth: 280,
            }}
          >
            {/* Loading indicator */}
            {variant === 'circular' && (
              <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                <CircularProgress 
                  size={sizeMap[size]} 
                  thickness={4}
                  sx={{
                    color: theme.palette.primary.main,
                  }}
                />
                {progress !== undefined && (
                  <Box
                    sx={{
                      top: 0,
                      left: 0,
                      bottom: 0,
                      right: 0,
                      position: 'absolute',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Typography
                      variant="caption"
                      component="div"
                      color="text.secondary"
                      sx={{ fontWeight: 600 }}
                    >
                      {Math.round(progress)}%
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            {/* Loading message */}
            <Box sx={{ textAlign: 'center' }}>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 500,
                  color: theme.palette.text.primary,
                  marginBottom: 1,
                }}
              >
                {message}
              </Typography>
              
              {progress !== undefined && variant === 'linear' && (
                <Box sx={{ width: 200, marginTop: 2 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={progress}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      backgroundColor: alpha(theme.palette.primary.main, 0.1),
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 3,
                        backgroundColor: theme.palette.primary.main,
                      },
                    }}
                  />
                  <Typography 
                    variant="body2" 
                    color="text.secondary"
                    sx={{ marginTop: 1, fontWeight: 500 }}
                  >
                    {Math.round(progress)}% complete
                  </Typography>
                </Box>
              )}
            </Box>

            {/* Loading dots animation */}
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              {[0, 1, 2].map((index) => (
                <motion.div
                  key={index}
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 1, 0.5],
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    delay: index * 0.2,
                    ease: "easeInOut",
                  }}
                >
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: theme.palette.primary.main,
                    }}
                  />
                </motion.div>
              ))}
            </Box>
          </Box>
        </motion.div>
      </Box>
    </motion.div>
  );
};