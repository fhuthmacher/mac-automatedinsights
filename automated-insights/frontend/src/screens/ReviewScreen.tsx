import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { generateClient } from 'aws-amplify/data';
import { getConversation } from '../graphql/queries';
import { recieveMessageChunkAsync } from "../graphql/subscriptions";
import { createMessageAsync } from '../graphql/mutations';

import {
  Box,
  Container,
  Paper,
  Typography,
  CircularProgress,
  useTheme,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Button,
  TextField,
  IconButton,
  Stack,
} from '@mui/material';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';




interface UseCase {
  name: string;
  description: string;
  modelDetails: ModelDetails;
  predictionDataLocation: string;
  dataScientistCommentary?: string;
}

interface TargetColumn {
  name: string;
  definition: string;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface ModelDetails {
  trainingDataLocation: string;
  modelLocation: string;
  mlDatasetSqlQuery?: string;
  testDataLocation?: string;
  mlDatasetLocation?: string;
  targetColumn?: TargetColumn;
  accuracy: number | string;
  featureImportance: Array<{
    feature: string;
    importance: number;
  }>;
  featureImportances?: string;
  samplePredictedValues: string[];
}


interface ResultBoxProps {
  title: string;
  content: string | React.ReactNode;
  isLast?: boolean;
}

interface Message {
  __typename: "Message";
  sender: string;
  message: string;
  createdAt: string;
}

interface Feedback {
  isPositive: boolean;
  comment: string;
}

const ResultBox = ({ title, content, isLast = false }: ResultBoxProps) => {
  const theme = useTheme();
  
  return (
    <Box sx={{ position: 'relative', mb: isLast ? 0 : 4 }}>
      <Paper 
        elevation={3}
        sx={{
          p: 4,
          borderRadius: 2,
          backgroundColor: '#fff',
        }}
      >
        <Typography variant="h4" gutterBottom sx={{ color: theme.palette.primary.main }}>
          {title}
        </Typography>
        {typeof content === 'string' ? (
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
            {content}
          </Typography>
        ) : (
          <Box sx={{ mt: 2 }}>
            {content}
          </Box>
        )}
      </Paper>
      {!isLast && (
        <Box
          sx={{
            position: 'absolute',
            bottom: '-32px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '2px',
            height: '32px',
            backgroundColor: theme.palette.primary.main,
            '&::after': {
              content: '""',
              position: 'absolute',
              bottom: '-8px',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '8px solid transparent',
              borderRight: '8px solid transparent',
              borderTop: `8px solid ${theme.palette.primary.main}`,
            }
          }}
        />
      )}
    </Box>
  );
};

const FeatureImportanceChart = ({ features }: { features: FeatureImportance[] }) => {
  return (
    <TableContainer>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 'bold' }}>Feature</TableCell>
            <TableCell sx={{ fontWeight: 'bold' }}>Importance (%)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {features.map((feature, index) => (
            <TableRow key={index}>
              <TableCell>{feature.feature}</TableCell>
              <TableCell>{feature.importance.toFixed(2)}%</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};


const ModelDetailsContent = ({ modelDetails, predictionDataLocation, dataScientistCommentary }: { modelDetails: ModelDetails; predictionDataLocation: string; dataScientistCommentary?: string }) => {
  // Improved accuracy display logic
  let accuracyDisplay = "";
  if (typeof modelDetails.accuracy === 'number') {
    // If it's a number between 0 and 1, treat as decimal (e.g., 0.95)
    if (modelDetails.accuracy >= 0 && modelDetails.accuracy <= 1) {
      accuracyDisplay = `${(modelDetails.accuracy * 100).toFixed(2)}%`;
    } 
    // If it's a number > 1 and < 100, assume it's already a percentage value
    else if (modelDetails.accuracy > 1 && modelDetails.accuracy <= 100) {
      accuracyDisplay = `${modelDetails.accuracy.toFixed(2)}%`;
    }
    // If it's a very large number, something might be wrong with the data
    else {
      accuracyDisplay = `${modelDetails.accuracy}`;
    }
  } else {
    // If it's a string, use it directly (might already include % sign)
    accuracyDisplay = modelDetails.accuracy.toString();
    // Add % sign if it doesn't already have one and appears to be a number
    if (!accuracyDisplay.includes('%') && !isNaN(parseFloat(accuracyDisplay))) {
      accuracyDisplay += '%';
    }
  }

  return (
    <Box>
      {modelDetails.targetColumn && (
        <>
          <Typography variant="h6" gutterBottom>
            Target Column
          </Typography>
          <Typography variant="body1" sx={{ fontWeight: 'bold', mb: 1 }}>
            {modelDetails.targetColumn.name}
          </Typography>
          <Typography variant="body2" sx={{ mb: 3 }}>
            {modelDetails.targetColumn.definition}
          </Typography>
        </>
      )}

      <Typography variant="h6" gutterBottom>
        Model Performance
      </Typography>
      <Typography variant="body1" sx={{ mb: 2 }}>
        Accuracy: {accuracyDisplay}
      </Typography>

      <Typography variant="h6" gutterBottom>
        Feature Importance
      </Typography>
      <Box sx={{ mb: 3 }}>
        {modelDetails.featureImportance && modelDetails.featureImportance.length > 0 ? (
          <FeatureImportanceChart features={modelDetails.featureImportance} />
        ) : (
          <Typography variant="body2">
            {modelDetails.featureImportances || "No feature importance data available"}
          </Typography>
        )}
      </Box>

      {dataScientistCommentary && (
        <>
          <Typography variant="h6" gutterBottom>
            Data Scientist Commentary
          </Typography>
          <Paper sx={{ p: 2, backgroundColor: 'rgba(25, 118, 210, 0.05)', border: '1px solid rgba(25, 118, 210, 0.2)', mb: 3 }}>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {dataScientistCommentary}
            </Typography>
          </Paper>
        </>
      )}

      <Typography variant="h6" gutterBottom>
        Training Data Location
      </Typography>
      <Typography variant="body2" sx={{ mb: 3 }}>
        {modelDetails.trainingDataLocation}
      </Typography>

      {modelDetails.testDataLocation && (
        <>
          <Typography variant="h6" gutterBottom>
            Test Data Location
          </Typography>
          <Typography variant="body2" sx={{ mb: 3 }}>
            {modelDetails.testDataLocation}
          </Typography>
        </>
      )}

      {modelDetails.mlDatasetLocation && (
        <>
          <Typography variant="h6" gutterBottom>
            ML Dataset Location
          </Typography>
          <Typography variant="body2" sx={{ mb: 3 }}>
            {modelDetails.mlDatasetLocation}
          </Typography>
        </>
      )}

      <Typography variant="h6" gutterBottom>
        Model Location
      </Typography>
      <Typography variant="body2" sx={{ mb: 3 }}>
        {modelDetails.modelLocation}
      </Typography>

      <Typography variant="h6" gutterBottom>
        Predictions Data Location
      </Typography>
      <Typography variant="body2" sx={{ mb: 3 }}>
        {predictionDataLocation}
      </Typography>

      <Typography variant="h6" gutterBottom>
        Sample Predictions
      </Typography>
      {modelDetails.samplePredictedValues.map((value, index) => (
        <Typography key={index} variant="body2" sx={{ mb: 1 }}>
          {value.includes(':') ? value : `Value ${index + 1}: ${value}`}
        </Typography>
      ))}

      {modelDetails.mlDatasetSqlQuery && (
        <>
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            ML Dataset SQL Query
          </Typography>
          <Paper sx={{ p: 2, backgroundColor: 'grey.100' }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {modelDetails.mlDatasetSqlQuery}
            </Typography>
          </Paper>
        </>
      )}
    </Box>
  );
};

const parseXMLResponse = (xmlString: string): { useCases: UseCase[] } => {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlString, "text/xml");

  const useCases = Array.from(xmlDoc.querySelectorAll("UseCase")).map((useCase) => {
    // Get basic use case info
    const name = useCase.querySelector("Name")?.textContent || "";
    const description = useCase.querySelector("Description")?.textContent || "";
    
    // Get model details
    const modelDetails = useCase.querySelector("ModelDetails");
    const trainingDataLocation = modelDetails?.querySelector("TrainingDataLocation")?.textContent || "";
    const modelLocation = modelDetails?.querySelector("ModelLocation")?.textContent || "";
    
    // Only keep mlDatasetSqlQuery
    const mlDatasetSqlQuery = modelDetails?.querySelector("MLDatasetSQLQuery")?.textContent?.trim() || "";
    
    // New fields
    const testDataLocation = modelDetails?.querySelector("TestDataLocation")?.textContent || "";
    const mlDatasetLocation = modelDetails?.querySelector("MLDatasetLocation")?.textContent || "";
    
    // Handle target column
    const targetColumnElement = modelDetails?.querySelector("TargetColumn");
    const targetColumn = targetColumnElement ? {
      name: targetColumnElement.querySelector("Name")?.textContent || "",
      definition: targetColumnElement.querySelector("Definition")?.textContent || ""
    } : undefined;
    
    // Handle accuracy that might be a string
    const accuracyText = modelDetails?.querySelector("Accuracy")?.textContent || "0";
    let accuracy: number | string;
    
    // Check if the accuracy text already contains a percentage sign
    if (accuracyText.includes('%')) {
      // If it's already a percentage string, store it as is
      accuracy = accuracyText;
    } else {
      // Try to parse as a number
      const parsedAccuracy = parseFloat(accuracyText);
      if (!isNaN(parsedAccuracy)) {
        // If it's a valid number, check its magnitude
        if (parsedAccuracy > 1 && parsedAccuracy <= 100) {
          // If between 1 and 100, it's likely already a percentage value
          accuracy = parsedAccuracy;
        } else if (parsedAccuracy >= 0 && parsedAccuracy <= 1) {
          // If between 0 and 1, it's likely a decimal representing a percentage
          accuracy = parsedAccuracy;
        } else {
          // For any other values, just use as is
          accuracy = parsedAccuracy;
        }
      } else {
        // If not a valid number, use the text as is
        accuracy = accuracyText;
      }
    }
    
    // Handle feature importance - could be elements or a text string
    let featureImportance: Array<{ feature: string, importance: number }> = [];
    const featureImportances = modelDetails?.querySelector("FeatureImportances")?.textContent || "";
    
    if (modelDetails?.querySelector("FeatureImportance")) {
      featureImportance = Array.from(modelDetails?.querySelectorAll("FeatureImportance > Feature") || [])
        .map((feature, index) => {
          const importance = parseFloat(
            modelDetails?.querySelectorAll("FeatureImportance > Importance")[index]?.textContent || "0"
          );
          return {
            feature: feature.textContent || "",
            importance
          };
        });
    }

    // Get predicted values
    const samplePredictedValues = Array.from(
      modelDetails?.querySelectorAll("Sample-PredictedValues > *") || []
    ).map(value => value.textContent || "");
    
    // If no individual values, check if there's a text content
    const samplePredictedText = modelDetails?.querySelector("Sample-PredictedValues")?.textContent || "";
    const predictionValues = samplePredictedValues.length > 0 ? 
      samplePredictedValues : 
      (samplePredictedText ? [samplePredictedText] : []);

    const predictionDataLocation = useCase.querySelector("PredictionDataLocation")?.textContent || "";
    
    // Get data scientist commentary
    const dataScientistCommentary = useCase.querySelector("DataScientistCommentary")?.textContent || "";

    return {
      name,
      description,
      modelDetails: {
        trainingDataLocation,
        modelLocation,
        mlDatasetSqlQuery,
        testDataLocation,
        mlDatasetLocation,
        targetColumn,
        accuracy,
        featureImportance,
        featureImportances,
        samplePredictedValues: predictionValues
      },
      predictionDataLocation,
      dataScientistCommentary
    };
  });

  return { useCases };
};



// Move FeedbackSection outside main component
const FeedbackSection = ({ 
  feedback, 
  onFeedbackChange, 
  onFeedbackToggle, 
  onSubmit, 
  isSubmitting 
}: {
  feedback: Feedback;
  onFeedbackChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onFeedbackToggle: (isPositive: boolean) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
}) => (
  <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
    <Typography variant="h6" gutterBottom>
      Provide Feedback
    </Typography>
    <Stack spacing={2}>
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
        <IconButton 
          color={feedback.isPositive ? 'primary' : 'default'}
          onClick={() => onFeedbackToggle(true)}
        >
          <ThumbUpIcon />
        </IconButton>
        <IconButton 
          color={!feedback.isPositive ? 'primary' : 'default'}
          onClick={() => onFeedbackToggle(false)}
        >
          <ThumbDownIcon />
        </IconButton>
      </Box>
      <TextField
        fullWidth
        multiline
        rows={3}
        variant="outlined"
        placeholder="Share your thoughts about this ML use case..."
        value={feedback.comment}
        onChange={onFeedbackChange}
        disabled={isSubmitting}
      />
      <Button
        variant="contained"
        onClick={onSubmit}
        disabled={isSubmitting || !feedback.comment.trim()}
        sx={{ alignSelf: 'flex-start' }}
      >
        {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
      </Button>
    </Stack>
  </Paper>
);

const ReviewScreen = () => {
  const location = useLocation();
  const [loading, setLoading] = useState(true);
  const [lastMessage, setLastMessage] = useState<Message | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const subscriptionRef = useRef<any>(null);
  const setupTimeoutRef = useRef<any>(null);
  const conversationId = location.state?.uploadId;
  const client = useMemo(() => generateClient({
    authMode: "userPool",
  }), []);
  const [feedback, setFeedback] = useState<Feedback>({
    isPositive: true,
    comment: ''
  });
  const [isFeedbackSubmitting, setIsFeedbackSubmitting] = useState(false);

  // Reset state when conversation ID changes
  useEffect(() => {
    if (conversationId) {
      setLoading(true);
      setLastMessage(null);
      setIsComplete(false);
      setError(null);
    }
  }, [conversationId]);

  // Effect for initial state fetch
  useEffect(() => {
    let isSubscribed = true;

    const fetchInitialState = async () => {
      if (!conversationId) {
        setError("No conversation ID provided");
        setLoading(false);
        return;
      }

      try {
        console.log("[Initial State] Fetching for ID:", conversationId);
        const result = await client.graphql({
          query: getConversation,
          variables: { input: { conversationId } }
        });

        if (!isSubscribed) return;

        if (!result.data) {
          throw new Error('No data returned from query');
        }

        console.log("[Initial State] Received:", result.data);
        const conversation = result.data.getConversation;
        console.log("[Initial State] Status:", conversation?.status);
        
        const messages = conversation?.messages || [];
        if (messages.length > 0) {
          const lastMsg = messages[messages.length - 1];
          console.log("[Initial State] Last message:", lastMsg);
          setLastMessage(lastMsg);
          
          // Always keep loading true if status is not COMPLETE
          if (conversation?.status !== 'COMPLETE') {
            console.log("[Initial State] Conversation not complete, keeping loading state");
            setLoading(true);
          } else {
            console.log("[Initial State] Conversation complete, setting loading to false");
            setLoading(false);
            setIsComplete(true);
          }
        }

        if (conversation?.status === 'COMPLETE') {
          console.log("[Initial State] Conversation is complete");
          setIsComplete(true);
          setLoading(false);
        }
      } catch (err) {
        if (!isSubscribed) return;
        console.error("[Initial State] Error:", err);
        setError("Failed to load conversation");
        setLoading(false);
      }
    };

    fetchInitialState();

    return () => {
      isSubscribed = false;
    };
  }, [conversationId, client]);

  // Separate effect for subscription
  useEffect(() => {
    if (!conversationId) return;
    
    // Don't set up subscription if we already have a complete message
    if (lastMessage?.sender === 'Assistant' && isComplete) {
      console.log("[Subscription] Skipping - already have complete message");
      return;
    }

    let isActive = true;

    const setupSubscription = () => {
      if (!isActive) return;
      
      // Clear any existing subscription
      if (subscriptionRef.current) {
        console.log("[Subscription] Cleaning up existing subscription");
        subscriptionRef.current.unsubscribe();
        subscriptionRef.current = null;
      }

      try {
        console.log("[Subscription] Creating subscription object");
        const subscription = client
          .graphql({
            query: recieveMessageChunkAsync,
            variables: { input: { conversationId } }
          })
          .subscribe({
            next: (response) => {
              if (!isActive) {
                console.log("[Subscription] Skipping - component inactive");
                return;
              }
              console.log("[Subscription] Message received:", response);
              const messageChunk = response.data?.recieveMessageChunkAsync;
              
              if (!messageChunk) {
                console.log("[Subscription] No message chunk in response");
                return;
              }

              console.log("[Subscription] Processing chunk:", {
                type: messageChunk.chunkType,
                status: messageChunk.status,
                content: messageChunk.chunk
              });

              if (messageChunk.chunkType === 'text') {
                console.log("[Subscription] Processing text chunk");
                setLastMessage((prevMessage) => {
                  if (!isActive) return prevMessage;
                  const updatedMessage = {
                    __typename: "Message" as const,
                    sender: 'Assistant',
                    message: prevMessage?.sender === 'Assistant' 
                      ? (prevMessage.message || '') + (messageChunk.chunk || '')
                      : messageChunk.chunk || '',
                    createdAt: prevMessage?.createdAt || new Date().toISOString()
                  };
                  console.log("[Subscription] Updated message:", updatedMessage);
                  return updatedMessage;
                });
                // Only set loading to false if we've received actual content
                if (messageChunk.chunk && messageChunk.chunk.trim().length > 0) {
                  setLoading(false);
                }
              }
              
              if (messageChunk.chunkType === 'error') {
                console.error("[Subscription] Error chunk received:", messageChunk.chunk);
                setError(messageChunk.chunk || "An error occurred");
                setLoading(false);
              }

              if (messageChunk.chunkType === 'status' && messageChunk.status === 'COMPLETE') {
                console.log("[Subscription] Processing complete");
                setLoading(false);
                setIsComplete(true);
              }
            },
            error: (error) => {
              if (!isActive) return;
              console.error("[Subscription] Error in subscription:", error);
              setError("Error receiving updates");
              setLoading(false);
            },
            complete: () => {
              console.log("[Subscription] Subscription completed normally");
            }
          });

        subscriptionRef.current = subscription;
        console.log("[Subscription] Setup completed successfully");
      } catch (error) {
        console.error("[Subscription] Error during setup:", error);
        if (isActive) {
          setError("Failed to setup message updates");
          setLoading(false);
        }
      }
    };

    // Delay subscription setup to handle StrictMode double mount
    setupTimeoutRef.current = setTimeout(() => {
      if (isActive) {
        setupSubscription();
      }
    }, 100);

    return () => {
      console.log("[Subscription] Starting cleanup");
      isActive = false;
      
      if (setupTimeoutRef.current) {
        clearTimeout(setupTimeoutRef.current);
        setupTimeoutRef.current = null;
      }
      
      // Don't unsubscribe here - we'll let the subscription continue
      // even if this effect is cleaned up and re-run
      // This prevents the loading indicator from disappearing prematurely
    };
  }, [conversationId, client, isComplete]);

  const handleFeedbackSubmit = async () => {
    try {
      setIsFeedbackSubmitting(true);
      
      // const conversationId = await handleCreateConversation(
      //   `{ "action": "feedback", "feedbackType": ${feedback.isPositive ? "Positive" : "Negative"}, "usecase": "${lastMessage?.message}", "comment": "${feedback.comment}"}`
      // );

      await client.graphql({
        query: createMessageAsync,
        variables: {
          input: {
            conversationId: conversationId,
            prompt: JSON.stringify({
              action: 'feedback',
              feedbackType: feedback.isPositive ? 'positive' : 'negative',
              useCase: lastMessage?.message, // Store the use case being rated
              comment: feedback.comment
            })
          }
        }
      });

      // Clear feedback after submission
      setFeedback({ isPositive: true, comment: '' });
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setIsFeedbackSubmitting(false);
    }
  };

  const handleFeedbackChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setFeedback(prev => ({
      ...prev,
      comment: e.target.value
    }));
  };

  const handleFeedbackToggle = (isPositive: boolean) => {
    setFeedback(prev => ({
      ...prev,
      isPositive
    }));
  };

  if (!conversationId) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Paper sx={{ p: 3 }}>
            <Typography color="error">
              No conversation ID provided. Please start from the upload page.
            </Typography>
          </Paper>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Paper sx={{ p: 3 }}>
            <Typography color="error">
              {error}
            </Typography>
          </Paper>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      {loading && (
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center',
          alignItems: 'center',
          mt: 8,
          mb: 4
        }}>
          <CircularProgress size={40} />
        </Box>
      )}

      <Box sx={{ mt: 4, mb: 4 }}>
        {lastMessage && (
          <>            
            {(() => {
              try {
                const { useCases } = parseXMLResponse(lastMessage.message);
                return (
                  <>
                    {useCases.map((useCase, index) => (
                      <Box key={index} sx={{ mb: 4 }}>
                        {index > 0 && (
                          <Divider sx={{ my: 4 }} />
                        )}
                        <ResultBox 
                          title={`Use Case ${index + 1}: ${useCase.name}`}
                          content={
                            <Box>
                              <Typography variant="body1" sx={{ mb: 3 }}>
                                {useCase.description}
                              </Typography>
                              
                              <ModelDetailsContent 
                                modelDetails={useCase.modelDetails}
                                predictionDataLocation={useCase.predictionDataLocation}
                                dataScientistCommentary={useCase.dataScientistCommentary}
                              />
                            </Box>
                          }
                          isLast={index === useCases.length - 1}
                        />
                      </Box>
                    ))}
                  </>
                );
              } catch (error) {
                console.error('Error parsing response:', error);
                return (
                  <ResultBox 
                    title="Raw Response" 
                    content={lastMessage.message || 'Processing...'}
                    isLast={true}
                  />
                );
              }
            })()}

            {isComplete && (
              <Typography 
                variant="body2" 
                color="text.secondary" 
                sx={{ mt: 2, textAlign: 'center' }}
              >
                Analysis complete
              </Typography>
            )}
          </>
        )}
      </Box>
      {isComplete && (
        <FeedbackSection 
          feedback={feedback}
          onFeedbackChange={handleFeedbackChange}
          onFeedbackToggle={handleFeedbackToggle}
          onSubmit={handleFeedbackSubmit}
          isSubmitting={isFeedbackSubmitting}
        />
      )}
    </Container>
  );
};

export default ReviewScreen; 