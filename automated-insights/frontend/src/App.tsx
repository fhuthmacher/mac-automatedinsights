import { BrowserRouter, Routes, Route, Link, Navigate } from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { Authenticator } from "@aws-amplify/ui-react";
import "@aws-amplify/ui-react/styles.css";
import { Amplify } from "aws-amplify";
import logo from './assets/logo.png';
import UploadScreen from './screens/UploadScreen'
import DiscoverScreen from './screens/DiscoverScreen'

import * as CdkData from "./cdk-outputs.json";
import ReviewScreen from "./screens/ReviewScreen";

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: CdkData.GenAiStack.genaiaaUserPoolId,
      userPoolClientId: CdkData.GenAiStack.genaiaaUserPoolClientId,
      identityPoolId: CdkData.GenAiStack.genaiaaIdentityPoolId,
    }
  },
  Storage: {
    S3: {
      bucket: CdkData.GenAiStack.genaiaaUserUploadBucketName,
      region: CdkData.GenAiStack.genaiaaAwsRegion,
    }
  },
  API: {
    GraphQL: {
      defaultAuthMode: "userPool",
      region: CdkData.GenAiStack.genaiaaAwsRegion,
      endpoint: CdkData.GenAiStack.genaiaaGraphQLAPIURL
    }
  }
});

// Update the configuration logging to match the new names
console.log('Amplify Configuration:', {
  userPoolId: CdkData.GenAiStack.genaiaaUserPoolId,
  userPoolClientId: CdkData.GenAiStack.genaiaaUserPoolClientId,
  identityPoolId: CdkData.GenAiStack.genaiaaIdentityPoolId,
  bucket: CdkData.GenAiStack.genaiaaUserUploadBucketName,
  region: CdkData.GenAiStack.genaiaaAwsRegion,
});

const theme = createTheme({
  palette: {
    primary: {
      main: '#00B4D8', // Bright cyan/blue
      light: '#48CAE4',
      dark: '#0096C7',
    },
    secondary: {
      main: '#FF0080', // Vibrant pink/magenta
      light: '#FF4D94',
      dark: '#CC0066',
    },
    background: {
      default: '#03045E', // Deep blue background
      paper: '#FFFFFF',
    },
    text: {
      primary: '#03045E', // Deep blue text
      secondary: '#48CAE4', // Lighter blue text
    },
  },
  typography: {
    fontFamily: "'Inter', 'Source Sans Pro', sans-serif",
    h4: {
      fontWeight: 600,
      color: '#00B4D8', // Use primary color instead of gradient
    },
    h5: {
      fontWeight: 600,
      color: '#00B4D8', // Use primary color instead of gradient
    },
    h6: {
      fontWeight: 500,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#03045E', // Solid color instead of gradient
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        contained: {
          backgroundColor: '#00B4D8', // Solid primary color
          color: 'white',
          '&:hover': {
            backgroundColor: '#0096C7', // Darker shade on hover
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 4px 20px rgba(0, 180, 216, 0.1)',
        },
      },
    },
  },
});

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <Authenticator>
        {({ signOut, user }) => (
          <Main signOut={signOut} user={user} />
        )}
      </Authenticator>
    </ThemeProvider>
  );
}

function Main(props: { signOut: any; user: any }) {
  return (
    <BrowserRouter>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" color="primary" elevation={0}>
          <Container maxWidth="xl">
            <Toolbar disableGutters sx={{ minHeight: '80px' }}>
              <Box sx={{ display: "flex", alignItems: "center", width: '100%' }}>
                <Box sx={{ display: "flex", alignItems: "center" }}>
                  <Link to="/upload">
                    <img
                      src={logo}
                      alt="Logo"
                      style={{ height: "40px", marginRight: "32px", cursor: "pointer" }}
                    />
                  </Link>
                  <Button
                    component={Link}
                    to="/upload"
                    sx={{
                      color: "white",
                      fontSize: "1rem",
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      },
                    }}
                  >
                    PROCESS FILE
                  </Button>
                  <Button
                    component={Link}
                    to="/discover"
                    sx={{
                      color: "white",
                      fontSize: "1rem",
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      },
                    }}
                  >
                    DISCOVER
                  </Button>
                </Box>
                <Box sx={{ marginLeft: 'auto' }}>
                  <Button
                    variant="text"
                    color="inherit"
                    onClick={props.signOut}
                    sx={{
                      color: "white",
                      fontSize: "1rem",
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      },
                    }}
                  >
                    SIGN OUT
                  </Button>
                </Box>
              </Box>
            </Toolbar>
          </Container>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4 }}>
          <Routes>
            <Route path="/upload" element={<UploadScreen />} />
            <Route path="/review" element={<ReviewScreen />} />
            <Route path="/discover" element={<DiscoverScreen />} />
            <Route path="/" element={
              <Box sx={{ textAlign: "center", mt: 8 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                  Welcome to Automated Insights
                </Typography>
                <Typography variant="subtitle1" gutterBottom sx={{ color: 'white' }}>
                  Click "Process File" to get started
                </Typography>
              </Box>
            } />
            
            <Route path="" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Container>
      </Box>
    </BrowserRouter>
  );
}
