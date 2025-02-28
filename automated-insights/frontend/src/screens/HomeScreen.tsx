import { Container, Box, Typography, Grid, Button } from '@mui/material'
import { Link } from 'react-router-dom'

export default function HomeScreen() {
  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Welcome to Automated Insights
        </Typography>
        <Grid container spacing={3} justifyContent="center" sx={{ mt: 2 }}>
          <Grid item>
            <Button
              variant="contained"
              color="primary"
              component={Link}
              to="/upload"
            >
              Upload Data
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="contained"
              color="primary"
              component={Link}
              to="/discover"
            >
              Discover Use Cases
            </Button>
          </Grid>
        </Grid>
      </Box>
    </Container>
  )
} 