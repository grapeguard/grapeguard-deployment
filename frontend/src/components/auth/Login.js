// Multilingual Login.js
// src/components/auth/Login.js

import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton
} from '@mui/material';
import {
  Email as EmailIcon,
  Lock as LockIcon,
  Visibility,
  VisibilityOff,
  Agriculture as AgricultureIcon
} from '@mui/icons-material';
import { useAuth } from '../../context/AuthContext';
import { useTranslation } from '../../context/LanguageContext';
import LanguageSwitcher from '../LanguageSwitcher';

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const { t } = useTranslation();
  
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      console.log('🔐 Starting login process...');
      await login(formData.email, formData.password);
      console.log('🎉 Login completed, redirecting...');
      
      // Small delay to ensure auth state is updated
      setTimeout(() => {
        navigate('/dashboard');
      }, 500);
      
    } catch (error) {
      console.error('💥 Login failed:', error);
      setError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-green-50 flex items-center justify-center p-4" style={{
      background: 'linear-gradient(to bottom right, #faf5ff, #f0fdf4)',
      minHeight: '100vh'
    }}>
      {/* Language Switcher */}
      <Box position="absolute" top={16} right={16}>
        <LanguageSwitcher variant="compact" />
      </Box>

      <Container maxWidth="sm">
        <Paper elevation={8} style={{ padding: '2rem', borderRadius: '1rem' }}>
          <Box textAlign="center" mb={4}>
            <div style={{
              background: 'linear-gradient(to right, #8b5cf6, #22c55e)',
              width: '4rem',
              height: '4rem',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1rem'
            }}>
              <AgricultureIcon style={{ color: 'white', fontSize: '1.5rem' }} />
            </div>
            <Typography variant="h4" component="h1" className="font-bold text-gray-800 mb-2">
              GrapeGuard
            </Typography>
            <Typography variant="body1" color="textSecondary">
              {t('welcomeBack')}
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" style={{ marginBottom: '1rem' }}>
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <Box mb={3}>
              <TextField
                fullWidth
                label={t('emailAddress')}
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                required
                variant="outlined"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <EmailIcon color="action" />
                    </InputAdornment>
                  ),
                }}
              />
            </Box>

            <Box mb={3}>
              <TextField
                fullWidth
                label={t('password')}
                name="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={handleChange}
                required
                variant="outlined"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <LockIcon color="action" />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
            </Box>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              style={{
                padding: '0.75rem',
                marginTop: '1.5rem',
                background: 'linear-gradient(to right, #8b5cf6, #22c55e)',
                color: 'white'
              }}
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                t('signIn')
              )}
            </Button>
          </form>

          <Box textAlign="center" mt={4}>
            <Typography variant="body2" color="textSecondary">
              {t('dontHaveAccount')}{' '}
              <Link
                to="/register"
                style={{ color: '#7c3aed', fontWeight: 500, textDecoration: 'none' }}
              >
                {t('signUpHere')}
              </Link>
            </Typography>
          </Box>
        </Paper>
      </Container>
    </div>
  );
}