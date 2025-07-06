// Fixed LiveCameraFeed.js with Multilingual Support
// src/components/live-monitoring/LiveCameraFeed.js

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Button,
  CircularProgress,
  Alert,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Clear as ClearIcon,
  Psychology as AIIcon,
  Camera as CameraIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import { useTranslation } from '../../context/LanguageContext';
import GoogleDriveService from '../../services/GoogleDriveService';
import DetectronDiseaseService from '../../services/DetectronDiseaseService';
import EnhancedCameraCard from './EnhancedCameraCard';

export default function LiveCameraFeed() {
  const [driveService] = useState(new GoogleDriveService());
  const [detectionService] = useState(new DetectronDiseaseService());
  const { t, language, formatSensorValue } = useTranslation(); // Get all needed values from hook
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [error, setError] = useState(null);
  
  const [cameraResults, setCameraResults] = useState({
    camera1: null,
    camera2: null
  });
  
  const [detectionHistory, setDetectionHistory] = useState([]);

  // FIXED: Move disease name mapping outside component and use language parameter
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

  // Load Detectron2 model on component mount
  useEffect(() => {
    const initModel = async () => {
      try {
        console.log('🤖 Loading Detectron2 model...');
        setModelError(null);
        
        const loaded = await detectionService.loadModel();
        setIsModelLoaded(loaded);
        
        if (!loaded) {
          setModelError('Detectron2 server not running. Please start the server on localhost:5000');
          console.error('❌ Detectron2 model failed to load');
        } else {
          console.log('✅ Detectron2 model loaded successfully');
        }
        
      } catch (error) {
        console.error('❌ Model loading failed:', error);
        setModelError(`Model loading failed: ${error.message}`);
        setIsModelLoaded(false);
      }
    };

    initModel();
  }, [detectionService]);

  // Load detection history from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('liveDetectionHistory');
      if (saved) {
        const parsed = JSON.parse(saved);
        setDetectionHistory(parsed);
        console.log('📜 Loaded history:', parsed.length, 'items');
      }
    } catch (error) {
      console.error('Failed to load detection history:', error);
    }
  }, []);

  const saveToHistory = (results) => {
    try {
      const newHistoryItems = [];
      
      if (results.camera1) {
        newHistoryItems.push({
          ...results.camera1,
          historyId: `history_${results.camera1.id}_${Date.now()}`,
          driveUploadTime: results.camera1.imageData?.createdTime || results.camera1.timestamp,
          driveFileName: results.camera1.imageData?.name || `camera1_${Date.now()}.jpg`
        });
      }
      
      if (results.camera2) {
        newHistoryItems.push({
          ...results.camera2,
          historyId: `history_${results.camera2.id}_${Date.now() + 1}`,
          driveUploadTime: results.camera2.imageData?.createdTime || results.camera2.timestamp,
          driveFileName: results.camera2.imageData?.name || `camera2_${Date.now()}.jpg`
        });
      }
      
      const updatedHistory = [...newHistoryItems, ...detectionHistory];
      setDetectionHistory(updatedHistory);
      localStorage.setItem('liveDetectionHistory', JSON.stringify(updatedHistory));
      
      console.log('💾 Saved to history:', newHistoryItems.length, 'new items, total:', updatedHistory.length);
      
    } catch (error) {
      console.error('Failed to save detection history:', error);
    }
  };

  const deleteIndividualHistory = (historyId) => {
    const updatedHistory = detectionHistory.filter(item => 
      (item.historyId || item.id) !== historyId
    );
    setDetectionHistory(updatedHistory);
    localStorage.setItem('liveDetectionHistory', JSON.stringify(updatedHistory));
    console.log(`🗑️ Deleted history item: ${historyId}`);
  };

  const startAutoDetection = async () => {
    if (!isModelLoaded) {
      setError('Detectron2 model not loaded. Please start the server on localhost:5000 and refresh the page.');
      return;
    }

    if (modelError) {
      setError('Cannot process: ' + modelError);
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      console.log('🚀 Starting Auto Detection through Camera...');
      
      const images = await driveService.getLatestCameraImages();
      
      if (!images.camera1 && !images.camera2) {
        throw new Error('No camera images found. Make sure ESP32-CAM is uploading images.');
      }
      
      let camera1Result = null;
      if (images.camera1) {
        console.log(`🔬 Processing Camera 1: ${images.camera1.name}`);
        camera1Result = await processImageWithDetectron2(images.camera1, 1);
      }
      
      let camera2Result = null;
      if (images.camera2) {
        console.log(`🔬 Processing Camera 2: ${images.camera2.name}`);
        camera2Result = await processImageWithDetectron2(images.camera2, 2);
      }
      
      const results = {
        camera1: camera1Result,
        camera2: camera2Result
      };
      
      setCameraResults(results);
      saveToHistory(results);
      
      console.log('🎉 Auto detection completed successfully!');
      
    } catch (error) {
      console.error('❌ Auto detection failed:', error);
      setError(`Detection failed: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const processImageWithDetectron2 = async (imageData, cameraNumber) => {
    const startTime = Date.now();
    
    try {
      console.log(`🤖 Running REAL Detectron2 on Camera ${cameraNumber}...`);
      
      const imageDataUrl = await driveService.getImageAsDataUrl(imageData);
      const img = new Image();
      
      const detectionResult = await new Promise((resolve, reject) => {
        img.onload = async () => {
          try {
            console.log(`🔬 Image loaded, running Detectron2 prediction...`);
            const result = await detectionService.predict(img, true);
            console.log(`✅ Detectron2 result:`, result);
            resolve(result);
          } catch (error) {
            console.error('❌ Detectron2 prediction failed:', error);
            reject(error);
          }
        };
        
        img.onerror = () => reject(new Error('Image loading failed'));
        img.crossOrigin = 'anonymous';
        img.src = imageDataUrl;
      });

      const processingTime = Date.now() - startTime;
      
      const result = {
        id: `real_${imageData.id}_${Date.now()}`,
        camera: cameraNumber,
        imageName: imageData.name,
        imageData: imageData,
        originalImage: imageDataUrl,
        visualizationImage: detectionResult.visualizationImage || null,
        detection: detectionResult,
        timestamp: new Date().toISOString(),
        processingTime: processingTime,
        isRealDetectron2: true,
        modelType: 'Detectron2',
        driveUploadTime: imageData.createdTime,
        driveFileName: imageData.name
      };
      
      console.log(`🎯 Camera ${cameraNumber} processed in ${processingTime}ms:`, {
        disease: detectionResult.disease,
        confidence: detectionResult.confidence,
        severity: detectionResult.severity,
        driveUploadTime: imageData.createdTime
      });
      
      return result;
      
    } catch (error) {
      console.error(`❌ Processing failed for Camera ${cameraNumber}:`, error);
      throw error;
    }
  };

  const clearCurrentResults = () => {
    setCameraResults({ camera1: null, camera2: null });
    setError(null);
    console.log('🧹 Cleared current results (history preserved)');
  };

  const clearHistory = () => {
    setDetectionHistory([]);
    localStorage.removeItem('liveDetectionHistory');
    console.log('🗑️ Cleared detection history');
  };

  const downloadResult = (result) => {
    if (result?.visualizationImage) {
      const link = document.createElement('a');
      link.href = result.visualizationImage;
      link.download = `detectron2-camera${result.camera}-${Date.now()}.jpg`;
      link.click();
    }
  };

  // Format Drive upload time with date
  const formatDriveUploadTime = (driveTime) => {
    if (!driveTime) return 'Unknown time';
    
    try {
      const date = new Date(driveTime);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (error) {
      return 'Invalid date';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Control Panel */}
      <Box display="flex" justifyContent="space-between" alignItems="center" gap={2}>
        <Box display="flex" gap={2}>
          <Button
            variant="contained"
            size="large"
            startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
            onClick={startAutoDetection}
            disabled={isProcessing || !isModelLoaded || modelError}
            style={{ 
              backgroundColor: (!isModelLoaded || modelError) ? '#9ca3af' : '#22c55e', 
              color: 'white',
              padding: '0.75rem 2rem'
            }}
          >
            {isProcessing ? t('processing') : t('autoDetectThroughCamera')}
          </Button>
          
          <Button
            variant="outlined"
            onClick={clearCurrentResults}
            disabled={isProcessing}
            startIcon={<ClearIcon />}
            style={{ borderColor: '#6b7280', color: '#6b7280' }}
            size="large"
          >
            {t('clearResults')}
          </Button>
        </Box>

        <Box display="flex" alignItems="center" gap={1}>
          <Chip
            label={t('googleDriveConnected')}
            color="success"
            icon={<CheckCircleIcon />}
            size="small"
          />
          <Chip
            label={isModelLoaded ? t('aiModelReady') : (modelError ? t('aiModelFailed') : t('aiModelLoading'))}
            color={isModelLoaded ? 'success' : (modelError ? 'error' : 'default')}
            icon={isModelLoaded ? <AIIcon /> : (modelError ? <ErrorIcon /> : <CircularProgress size={16} />)}
            size="small"
          />
        </Box>
      </Box>

      {/* Error Display */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Model Error Display */}
      {modelError && (
        <Alert severity="error" style={{ backgroundColor: '#fef2f2', border: '1px solid #fca5a5' }}>
          <Typography variant="body2">
            <strong>{t('detectronServerError')}:</strong> {modelError}
          </Typography>
          <Typography variant="body2" style={{ marginTop: '0.5rem' }}>
            Please start the server: <code>python detectron_server.py</code>
          </Typography>
        </Alert>
      )}

      {/* Camera Results */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <EnhancedCameraCard 
            cameraData={cameraResults.camera1}
            cameraNumber={1}
            onCameraClick={(data) => console.log('Camera 1 details:', data)}
            onDownload={downloadResult}
          />
        </Grid>
        <Grid item xs={12} md={6}>
          <EnhancedCameraCard 
            cameraData={cameraResults.camera2}
            cameraNumber={2}
            onCameraClick={(data) => console.log('Camera 2 details:', data)}
            onDownload={downloadResult}
          />
        </Grid>
      </Grid>

      {/* Detection History - Show exactly 5 per row */}
      <Card elevation={2} style={{ backgroundColor: 'white', borderRadius: '12px' }}>
        <CardContent style={{ padding: '1.5rem' }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" style={{ fontWeight: 600 }}>
              {t('recentAnalysisHistory')} ({detectionHistory.length})
            </Typography>
            
            {detectionHistory.length > 0 && (
              <Button
                onClick={clearHistory}
                style={{ color: '#ef4444' }}
                size="small"
                startIcon={<DeleteIcon />}
              >
                {t('clearHistory')}
              </Button>
            )}
          </Box>
          
          {detectionHistory.length === 0 ? (
            <Box textAlign="center" py={3} style={{ backgroundColor: '#f9fafb', borderRadius: '8px' }}>
              <CameraIcon style={{ fontSize: '3rem', color: '#9ca3af', marginBottom: '1rem' }} />
              <Typography variant="h6" color="textSecondary" style={{ marginBottom: '0.5rem' }}>
                {t('noAIDetectionsYet')}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {t('clickAutoDetect')}
              </Typography>
            </Box>
          ) : (
            <Grid container spacing={2}>
              {/* Show exactly 5 items per row - 2.4 columns each (12/5 = 2.4) */}
              {detectionHistory.map((result) => {
                // FIXED: Get disease names here using current language
                const diseaseNames = getDiseaseDisplayName(result.detection.disease, language);
                
                return (
                  <Grid 
                    item 
                    xs={12}    // Full width on mobile
                    sm={6}     // 2 per row on small screens
                    md={2.4}   // 5 per row on medium+ screens (12/5 = 2.4)
                    key={result.historyId || result.id}
                  >
                    <Card 
                      elevation={1} 
                      style={{ 
                        transition: 'all 0.3s ease',
                        border: '1px solid #e5e7eb',
                        position: 'relative',
                        backgroundColor: 'white',
                        height: '280px' // Fixed height for consistency
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-2px)';
                        e.currentTarget.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
                      }}
                    >
                      {/* Individual Delete Button */}
                      <Box
                        style={{
                          position: 'absolute',
                          top: '0.5rem',
                          right: '0.5rem',
                          zIndex: 1
                        }}
                      >
                        <Tooltip title={t('deleteThisAnalysis')}>
                          <IconButton
                            size="small"
                            onClick={() => deleteIndividualHistory(result.historyId || result.id)}
                            style={{ 
                              backgroundColor: 'rgba(239, 68, 68, 0.9)',
                              color: 'white',
                              width: '24px',
                              height: '24px'
                            }}
                          >
                            <DeleteIcon style={{ fontSize: '0.75rem' }} />
                          </IconButton>
                        </Tooltip>
                      </Box>

                      <CardContent style={{ padding: '1rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
                        {/* Image Preview Box */}
                        <Box 
                          style={{
                            width: '100%',
                            height: '120px',
                            backgroundColor: '#f3f4f6',
                            borderRadius: '6px',
                            marginBottom: '0.5rem',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            overflow: 'hidden'
                          }}
                        >
                          {result.visualizationImage ? (
                            <img
                              src={result.visualizationImage}
                              alt={`${result.detection.disease} detection`}
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                              }}
                            />
                          ) : (
                            <Box textAlign="center">
                              <CameraIcon style={{ fontSize: '2rem', color: '#9ca3af', marginBottom: '0.5rem' }} />
                              <Typography variant="caption" color="textSecondary">
                                {t('analysisResult')}
                              </Typography>
                            </Box>
                          )}
                        </Box>
                        
                        {/* FIXED: Disease name with proper language display */}
                        <Typography variant="body2" style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                          {diseaseNames.primary}
                        </Typography>
                        {diseaseNames.secondary && diseaseNames.secondary !== diseaseNames.primary && (
                          <Typography variant="caption" color="textSecondary" style={{ display: 'block', marginBottom: '0.25rem', fontStyle: 'italic' }}>
                            {diseaseNames.secondary}
                          </Typography>
                        )}
                        
                        <Typography variant="caption" color="textSecondary" style={{ display: 'block', marginBottom: '0.25rem' }}>
                          {t('camera')} {result.camera} • {formatSensorValue(result.detection.confidence, 1)}% {t('confidence')}
                        </Typography>
                        
                        {/* Show Drive upload date/time + Analysis time */}
                        <Typography variant="caption" color="textSecondary" style={{ display: 'block', marginBottom: '0.5rem', fontStyle: 'italic' }}>
                          📸 {t('drive')}: {formatDriveUploadTime(result.driveUploadTime || result.imageData?.createdTime)}
                          <br />
                          🤖 {t('analyzed')}: {new Date(result.timestamp).toLocaleDateString()} {new Date(result.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </Typography>
                        
                        <Box display="flex" gap={0.5} flexWrap="wrap" style={{ marginTop: 'auto' }}>
                          <Chip
                            label={result.detection.severity}
                            size="small"
                            color={
                              result.detection.severity === 'High' ? 'error' :
                              result.detection.severity === 'Medium' ? 'warning' :
                              result.detection.severity === 'None' ? 'success' : 'default'
                            }
                          />
                          {result.detection.detectedRegions > 0 && (
                            <Chip
                              label={`${formatSensorValue(result.detection.detectedRegions, 0)} ${t('regions')}`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                          <Chip
                            label="AI"
                            size="small"
                            variant="outlined"
                            style={{ fontSize: '0.7rem' }}
                          />
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          )}
        </CardContent>
      </Card>
    </div>
  );
}