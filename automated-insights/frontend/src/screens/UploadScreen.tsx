import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { uploadData } from 'aws-amplify/storage';
import { generateClient } from 'aws-amplify/data';
import { createConversation, createMessageAsync } from '../graphql/mutations';

interface FileType extends File {
  name: string;
}

const client = generateClient({
  authMode: "userPool",
});

const UploadScreen = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const [files, setFiles] = useState<FileType[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files) as FileType[]);
      setError(null);
    }
  };

  const uploadToS3 = async (file: File) => {
    try {
      const result = await uploadData({
        key: `${file.name}`,
        data: file,
        options: {
          contentType: file.type
        }
      }).result;
      
      return result.key;
    } catch (err) {
      console.error('Error uploading file:', err);
      throw err;
    }
  };

  const handleCreateConversation = async (text: string) => {
    try {
      const res = await client.graphql({
        query: createConversation,
      });
      
      const conversation = res.data?.createConversation?.conversation;
      
      if (!conversation?.conversationId) {
        throw new Error('Failed to create conversation: Invalid response');
      }
      
      await client.graphql({
        query: createMessageAsync,
        variables: {
          input: {
            conversationId: conversation.conversationId,
            prompt: JSON.stringify({
              action: 'upload',
              files: text
            })
          }
        }
      });

      return conversation.conversationId;
    } catch (err) {
      console.error('Error creating conversation:', err);
      throw err;
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select files to upload');
      return;
    }

    setUploading(true);
    try {
      const uploadPromises = files.map(file => uploadToS3(file));
      const uploadedKeys = await Promise.all(uploadPromises);
      
      const conversationId = await handleCreateConversation(
        `Processing files: ${uploadedKeys.join(', ')}`
      );

      navigate('/review', { 
        state: { 
          uploadId: conversationId,
          fileKeys: uploadedKeys,
          fromUpload: true
        },
        replace: true
      });
    } catch (err) {
      console.error('Error processing data:', err);
      setError('Failed to process the data. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ px: isMobile ? 2 : 3 }}>
      <Box sx={{ 
        mt: isMobile ? 2 : 4,
        mb: isMobile ? 2 : 4 
      }}>
        <Paper sx={{ 
          p: isMobile ? 2 : 3,
          borderRadius: 4,
          background: 'linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(240,249,255,1) 100%)',
        }}>
          <Typography 
            variant={isMobile ? "h5" : "h4"} 
            gutterBottom
            sx={{
              textAlign: isMobile ? 'center' : 'left',
              mb: isMobile ? 2 : 3
            }}
          >
            Upload Files
          </Typography>

          <Typography 
            variant={isMobile ? "subtitle1" : "h6"} 
            gutterBottom
            sx={{ mb: isMobile ? 1 : 2 }}
          >
            Select Files to Process
          </Typography>
          
          <Box
            sx={{
              border: '2px dashed',
              borderColor: 'primary.light',
              borderRadius: 3,
              p: isMobile ? 2 : 3,
              textAlign: 'center',
              mb: 3,
              cursor: 'pointer',
              minHeight: isMobile ? '150px' : '200px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              backgroundColor: 'rgba(0, 180, 216, 0.05)',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'secondary.main',
                backgroundColor: 'rgba(255, 0, 128, 0.05)',
                transform: 'translateY(-2px)',
              },
            }}
            component="label"
          >
            <input
              type="file"
              multiple
              accept=".csv,.json,.zip"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <CloudUploadIcon sx={{ 
              fontSize: isMobile ? 48 : 64, 
              color: 'primary.main', 
              mb: isMobile ? 1 : 2 
            }} />
            <Typography 
              variant={isMobile ? "body1" : "h6"} 
              sx={{ mb: 1 }}
            >
              {isMobile ? 'Tap to select files' : 'Drag and drop files here or click to select'}
            </Typography>
            <Typography 
              variant="body2" 
              color="textSecondary"
              sx={{ 
                fontSize: isMobile ? '0.75rem' : '0.875rem',
                px: isMobile ? 2 : 0
              }}
            >
              Supported formats: CSV, JSON, ZIP
            </Typography>
          </Box>

          {files.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography 
                variant={isMobile ? "body1" : "subtitle1"}
                sx={{ mb: 0.5 }}
              >
                Selected files ({files.length}):
              </Typography>
              <Box sx={{ 
                maxHeight: isMobile ? '100px' : '150px', 
                overflowY: 'auto',
                px: 1 
              }}>
                {files.map((file, index) => (
                  <Typography 
                    key={index} 
                    variant="body2" 
                    color="textSecondary"
                    sx={{ 
                      fontSize: isMobile ? '0.75rem' : '0.875rem',
                      mb: 0.5
                    }}
                  >
                    {file.name}
                  </Typography>
                ))}
              </Box>
            </Box>
          )}

          {error && (
            <Alert 
              severity="error" 
              sx={{ 
                mb: 2,
                fontSize: isMobile ? '0.75rem' : '0.875rem' 
              }}
            >
              {error}
            </Alert>
          )}

          {uploading && <LinearProgress sx={{ mb: 2 }} />}

          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={uploading}
            fullWidth
            sx={{
              py: isMobile ? 1.5 : 2,
              fontSize: isMobile ? '1rem' : '1.1rem',
              borderRadius: 2,
              transition: 'transform 0.2s ease',
              '&:not(:disabled):hover': {
                transform: 'translateY(-2px)',
              },
            }}
          >
            {uploading ? 'Processing...' : 'Process Files'}
          </Button>
        </Paper>
      </Box>
    </Container>
  );
};

export default UploadScreen; 