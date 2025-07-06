// Updated EnhancedCameraCard.js with Complete Multilingual Support
// src/components/live-monitoring/EnhancedCameraCard.js

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Camera as CameraIcon,
  Download as DownloadIcon,
  Compare as CompareIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Psychology as AIIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { useTranslation } from '../../context/LanguageContext';

export default function EnhancedCameraCard({ 
  cameraData, 
  cameraNumber, 
  onCameraClick, 
  onDownload 
}) {
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const { t, formatSensorValue, language, formatTimeAgo, getSeverityLabel } = useTranslation();

  // FIXED: Disease name mapping with proper language handling
  const getDiseaseDisplayName = (disease, currentLanguage) => {
    const diseaseMap = {
      'Karpa (Anthracnose)': {
        english: 'Karpa (Anthracnose)',
        marathi: 'कर्पा रोग',
        hindi: 'कर्पा (एंथ्रैक्नोस)'
      },
      'Bhuri (Powdery Mildew)': {
        english: 'Bhuri (Powdery Mildew)',
        marathi: 'भुरी रोग',
        hindi: 'भुरी (पाउडरी मिल्ड्यू)'
      },
      'Bokadlela (Borer Infestation)': {
        english: 'Bokadlela (Borer Infestation)',
        marathi: 'बोकाडलेला',
        hindi: 'बोकाडलेला (बोरर संक्रमण)'
      },
      'Davnya (Downy Mildew)': {
        english: 'Davnya (Downy Mildew)',
        marathi: 'दवयाचा रोग',
        hindi: 'दवयाचा (डाउनी मिल्ड्यू)'
      },
      'Healthy': {
        english: 'Healthy',
        marathi: 'निरोगी',
        hindi: 'स्वस्थ'
      }
    };

    const diseaseInfo = diseaseMap[disease] || { 
      english: disease, 
      marathi: disease, 
      hindi: disease 
    };

    if (currentLanguage === 'marathi') {
      return {
        primary: diseaseInfo.marathi,
        secondary: diseaseInfo.english
      };
    } else if (currentLanguage === 'hindi') {
      return {
        primary: diseaseInfo.hindi,
        secondary: diseaseInfo.english
      };
    } else {
      return {
        primary: diseaseInfo.english,
        secondary: null
      };
    }
  };

  const handleImageLoad = () => {
    setImageLoading(false);
    setImageError(false);
  };

  const handleImageError = () => {
    setImageLoading(false);
    setImageError(true);
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'High':
        return <ErrorIcon style={{ color: '#ef4444' }} />;
      case 'Medium':
        return <WarningIcon style={{ color: '#f59e0b' }} />;
      case 'None':
        return <CheckCircleIcon style={{ color: '#22c55e' }} />;
      default:
        return <AIIcon style={{ color: '#6b7280' }} />;
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'High':
        return '#ef4444';
      case 'Medium':
        return '#f59e0b';
      case 'None':
        return '#22c55e';
      default:
        return '#6b7280';
    }
  };

  // No data state
  if (!cameraData) {
    return (
      <Card 
        elevation={2} 
        style={{ 
          height: '400px', 
          border: '2px dashed #d1d5db', 
          borderRadius: '12px',
          backgroundColor: 'white'
        }}
      >
        <CardContent style={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'center', 
          alignItems: 'center',
          padding: '2rem'
        }}>
          <CameraIcon style={{ fontSize: '4rem', color: '#9ca3af', marginBottom: '1rem' }} />
          
          <Typography variant="h6" style={{ fontWeight: 600, marginBottom: '0.5rem', color: '#1f2937' }}>
            {t('camera')} {cameraNumber}
          </Typography>
          
          <Typography variant="body2" color="textSecondary" style={{ textAlign: 'center', marginBottom: '1rem' }}>
            {t('waitingForAIDetection')}
          </Typography>
          
          <Chip
            label={t('readyForProcessing')}
            size="small"
            style={{ 
              backgroundColor: '#fef3c7', 
              color: '#92400e',
              fontWeight: 600
            }}
            icon={<AIIcon style={{ color: '#92400e' }} />}
          />
        </CardContent>
      </Card>
    );
  }

  // FIXED: Get disease names for display using current language
  const diseaseNames = getDiseaseDisplayName(cameraData.detection?.disease || 'Unknown', language);

  // Results Display - Enhanced with proper multilingual support
  return (
    <>
      <Card 
        elevation={2} 
        style={{ 
          height: '500px', 
          borderRadius: '12px',
          backgroundColor: 'white',
          border: `2px solid ${getSeverityColor(cameraData.detection?.severity)}`,
          transition: 'all 0.3s ease'
        }}
      >
        <CardContent style={{ padding: '1rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Header with Disease Info */}
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Box display="flex" alignItems="center" gap={1}>
              <CameraIcon style={{ 
                color: cameraNumber === 1 ? '#22c55e' : '#8b5cf6', 
                fontSize: '1.25rem' 
              }} />
              <Typography variant="h6" style={{ fontWeight: 600, color: '#1f2937' }}>
                {t('camera')} {cameraNumber}
              </Typography>
              <AIIcon style={{ color: '#8b5cf6', fontSize: '1rem' }} />
            </Box>
            
            <Box display="flex" alignItems="center" gap={0.5}>
              {getSeverityIcon(cameraData.detection?.severity)}
              <Chip
                label={getSeverityLabel(cameraData.detection?.severity || 'Unknown')}
                size="small"
                style={{
                  backgroundColor: cameraData.detection?.severity === 'None' ? '#dcfce7' : 
                                  cameraData.detection?.severity === 'High' ? '#fecaca' : '#fef3c7',
                  color: getSeverityColor(cameraData.detection?.severity),
                  fontWeight: 600
                }}
              />
            </Box>
          </Box>

          {/* FIXED: Disease Detection Alert with proper language display */}
          <Box 
            style={{
              padding: '0.75rem',
              backgroundColor: cameraData.detection?.severity === 'None' ? '#f0fdf4' : 
                              cameraData.detection?.severity === 'High' ? '#fef2f2' : '#fffbeb',
              border: `1px solid ${getSeverityColor(cameraData.detection?.severity)}`,
              borderRadius: '8px',
              marginBottom: '1rem'
            }}
          >
            {/* FIXED: Primary language name (large) */}
            <Typography variant="subtitle1" style={{ fontWeight: 600, color: '#1f2937', marginBottom: '0.25rem' }}>
              {diseaseNames.primary}
            </Typography>
            
            {/* FIXED: Secondary language name (small) - only if different from primary */}
            {diseaseNames.secondary && diseaseNames.secondary !== diseaseNames.primary && (
              <Typography variant="body2" style={{ fontStyle: 'italic', color: '#6b7280', marginBottom: '0.5rem' }}>
                {diseaseNames.secondary}
              </Typography>
            )}
            
            <Typography variant="caption" color="textSecondary">
              {t('image')}: {cameraData.imageName} • {t('captured')}: {formatTimeAgo(cameraData.imageData?.createdTime)}
            </Typography>
          </Box>

          {/* AI Visualization */}
          <Box 
            style={{ 
              flex: 1, 
              position: 'relative',
              backgroundColor: '#f9fafb',
              borderRadius: '8px',
              overflow: 'hidden',
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            {imageLoading && (
              <Box 
                style={{ 
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  zIndex: 2
                }}
              >
                <CircularProgress />
              </Box>
            )}
            
            {imageError ? (
              <Box 
                style={{ 
                  height: '100%',
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  color: '#6b7280',
                  padding: '1rem'
                }}
              >
                <ErrorIcon style={{ fontSize: '2rem', marginBottom: '0.5rem', color: '#ef4444' }} />
                <Typography variant="body2" style={{ textAlign: 'center' }}>
                  AI visualization failed
                </Typography>
              </Box>
            ) : (
              <img
                src={cameraData.visualizationImage}
                alt={`${t('camera')} ${cameraNumber} AI Detection`}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: imageLoading ? 'none' : 'block'
                }}
                onLoad={handleImageLoad}
                onError={handleImageError}
                onLoadStart={() => setImageLoading(true)}
              />
            )}
            
            {/* Action buttons overlay */}
            {!imageError && (
              <Box
                style={{
                  position: 'absolute',
                  top: '0.5rem',
                  right: '0.5rem',
                  display: 'flex',
                  gap: '0.25rem'
                }}
              >
                <Tooltip title={t('compareImages')}>
                  <IconButton
                    size="small"
                    style={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.9)',
                      color: '#1f2937'
                    }}
                    onClick={() => setComparisonOpen(true)}
                  >
                    <CompareIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title={t('downloadResult')}>
                  <IconButton
                    size="small"
                    style={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.9)',
                      color: '#1f2937'
                    }}
                    onClick={() => onDownload && onDownload(cameraData)}
                  >
                    <DownloadIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            )}

            {/* AI Processing Indicator */}
            <Box
              style={{
                position: 'absolute',
                top: '0.5rem',
                left: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                backgroundColor: 'rgba(139, 92, 246, 0.9)',
                color: 'white',
                padding: '0.25rem 0.5rem',
                borderRadius: '12px',
                fontSize: '0.75rem',
                fontWeight: 600
              }}
            >
              <AIIcon style={{ fontSize: '0.75rem', marginRight: '0.25rem' }} />
              {t('aiDetected')}
            </Box>
          </Box>

          {/* FIXED: Statistics Grid with proper number formatting */}
          <Grid container spacing={1} mb={1}>
            <Grid item xs={4}>
              <Box style={{ 
                backgroundColor: cameraData.detection?.severity === 'None' ? '#f0fdf4' : '#fef2f2', 
                padding: '0.5rem', 
                borderRadius: '6px', 
                textAlign: 'center' 
              }}>
                <Typography variant="h6" style={{ 
                  fontWeight: 'bold', 
                  color: cameraData.detection?.severity === 'None' ? '#22c55e' : '#ef4444',
                  marginBottom: '0.25rem'
                }}>
                  {formatSensorValue(cameraData.detection?.detectedRegions || 0, 0)}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {t('affectedRegions')}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={4}>
              <Box style={{ 
                backgroundColor: '#f0fdf4', 
                padding: '0.5rem', 
                borderRadius: '6px', 
                textAlign: 'center' 
              }}>
                <Typography variant="h6" style={{ 
                  fontWeight: 'bold', 
                  color: '#22c55e',
                  marginBottom: '0.25rem'
                }}>
                  {formatSensorValue(cameraData.detection?.healthyArea || 0, 0)}%
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {t('healthyArea')}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={4}>
              <Box style={{ 
                backgroundColor: '#eff6ff', 
                padding: '0.5rem', 
                borderRadius: '6px', 
                textAlign: 'center' 
              }}>
                <Typography variant="h6" style={{ 
                  fontWeight: 'bold', 
                  color: '#3b82f6',
                  marginBottom: '0.25rem'
                }}>
                  {formatSensorValue(cameraData.detection?.confidence || 0, 1)}%
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {t('aiConfidence')}
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {/* FIXED: Processing Info with proper formatting and translations */}
          <Typography variant="caption" color="textSecondary" style={{ textAlign: 'center' }}>
            {t('processedIn')} {formatSensorValue(cameraData.processingTime || 0, 0)}ms • {t('realDetectron')} • {formatTimeAgo(cameraData.timestamp)}
          </Typography>
        </CardContent>
      </Card>

      {/* Comparison Dialog */}
      <Dialog
        open={comparisonOpen}
        onClose={() => setComparisonOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">
              {t('aiAnalysisComparison')} - {t('camera')} {cameraNumber}
            </Typography>
            <IconButton onClick={() => setComparisonOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                {t('originalImage')} {t('espCam')}
              </Typography>
              <img
                src={cameraData.originalImage}
                alt="Original"
                style={{
                  width: '100%',
                  maxHeight: '400px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb'
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                {t('aiDetectionResults')}
              </Typography>
              <img
                src={cameraData.visualizationImage}
                alt="AI Detection"
                style={{
                  width: '100%',
                  maxHeight: '400px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb'
                }}
              />
            </Grid>
          </Grid>
          
          {/* FIXED: Detection details with proper translations */}
          <Box mt={2} p={2} style={{ backgroundColor: '#f8fafc', borderRadius: '8px' }}>
            <Typography variant="subtitle2" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
              {t('detectionDetails')}
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>{t('disease')}:</strong> {diseaseNames.primary}
                </Typography>
                <Typography variant="body2">
                  <strong>{t('confidence')}:</strong> {formatSensorValue(cameraData.detection.confidence, 1)}%
                </Typography>
                <Typography variant="body2">
                  <strong>{t('severity')}:</strong> {getSeverityLabel(cameraData.detection.severity)}
                </Typography>
                <Typography variant="body2">
                  <strong>{t('regions')}:</strong> {formatSensorValue(cameraData.detection.detectedRegions, 0)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2">
                  <strong>{t('processingTime')}:</strong> {formatSensorValue(cameraData.processingTime || 0, 0)}ms
                </Typography>
                <Typography variant="body2">
                  <strong>{t('model')}:</strong> {t('realDetectron')}
                </Typography>
                <Typography variant="body2">
                  <strong>{t('source')}:</strong> {t('espCam')}
                </Typography>
                <Typography variant="body2">
                  <strong>{t('image')}:</strong> {cameraData.imageName}
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => onDownload && onDownload(cameraData)} 
            startIcon={<DownloadIcon />}
          >
            {t('downloadAnalysis')}
          </Button>
          <Button onClick={() => setComparisonOpen(false)}>
            {t('close')}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}