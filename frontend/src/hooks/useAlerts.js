// Complete useAlerts.js - Fixed Disease Name Display in Alerts
// src/hooks/useAlerts.js

import { useState, useEffect, useCallback } from 'react';
import { useSensorData } from './useSensorData';
import { useAuth } from '../context/AuthContext';

// Enhanced thresholds for sensor alerts
const SENSOR_THRESHOLDS = {
  temperature: { 
    optimal: [24, 34],
    warning: [20, 38],
    critical: [15, 42]
  },
  humidity: { 
    optimal: [40, 60],
    warning: [30, 85],
    critical: [20, 95]
  },
  soilMoisture: { 
    optimal: [30, 75],
    warning: [20, 85],
    critical: [10, 95]
  },
  lightIntensity: { 
    optimal: [200, 2000],
    warning: [100, 3000],
    critical: [50, 4000]
  },
  batteryVoltage: { 
    optimal: [9.5, 14.5],
    warning: [8.5, 15.0],
    critical: [7.0, 16.0]
  },
  rainSensor: { 
    optimal: [0, 0.1],
    warning: [0.1, 0.5],
    critical: [0.5, 2.0]
  }
};

export const useAlerts = () => {
  const { sensorData } = useSensorData();
  const { userProfile } = useAuth();
  
  // Three separate alert categories
  const [sensorAlerts, setSensorAlerts] = useState([]);
  const [manualDetectionAlerts, setManualDetectionAlerts] = useState([]);
  const [liveDetectionAlerts, setLiveDetectionAlerts] = useState([]);
  
  // Enhanced persistent preferences
  const [alertPreferences, setAlertPreferences] = useState(() => {
    try {
      const saved = localStorage.getItem('grapeGuardAlertPrefs_v8');
      if (saved) {
        const parsed = JSON.parse(saved);
        return {
          enableSensorAlerts: true,
          enableManualAlerts: true,
          enableLiveAlerts: true,
          criticalOnly: false,
          autoRefresh: true,
          dismissedAlerts: [],
          readAlerts: [],
          sectionsExpanded: {
            sensor: true,
            manual: true,
            live: true
          },
          lastUpdate: Date.now(),
          ...parsed
        };
      }
    } catch (error) {
      console.error('Failed to load alert preferences:', error);
    }
    
    return {
      enableSensorAlerts: true,
      enableManualAlerts: true,
      enableLiveAlerts: true,
      criticalOnly: false,
      autoRefresh: true,
      dismissedAlerts: [],
      readAlerts: [],
      sectionsExpanded: {
        sensor: true,
        manual: true,
        live: true
      },
      lastUpdate: Date.now()
    };
  });

  const userLocation = userProfile?.location || userProfile?.farmName || 'Western Maharashtra Vineyard';

  // Save preferences with debouncing
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      try {
        localStorage.setItem('grapeGuardAlertPrefs_v8', JSON.stringify({
          ...alertPreferences,
          lastUpdate: Date.now()
        }));
        console.log('💾 Alert preferences saved');
      } catch (error) {
        console.error('Failed to save alert preferences:', error);
      }
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [alertPreferences]);

  // Generate multilingual sensor alert titles and messages
  const generateSensorAlertData = useCallback((sensorType, value, alertType, optimal) => {
    const [minOptimal, maxOptimal] = optimal;

    const alertData = {
      temperature: {
        high: {
          titleKey: 'highTemperatureAlert',
          messageKey: 'temperatureExceedsOptimal'
        },
        low: {
          titleKey: 'lowTemperatureAlert', 
          messageKey: 'temperatureBelowOptimal'
        }
      },
      humidity: {
        high: {
          titleKey: 'highHumidityAlert',
          messageKey: 'humidityExceedsOptimal'
        },
        low: {
          titleKey: 'lowHumidityAlert',
          messageKey: 'humidityBelowOptimal'
        }
      },
      soilMoisture: {
        high: {
          titleKey: 'excessSoilMoisture',
          messageKey: 'soilMoistureExceedsOptimal'
        },
        low: {
          titleKey: 'lowSoilMoisture',
          messageKey: 'soilMoistureBelowOptimal'
        }
      },
      batteryVoltage: {
        low: {
          titleKey: 'lowBatteryVoltage',
          messageKey: 'batteryVoltageBelowOptimal'
        }
      },
      rainSensor: {
        high: {
          titleKey: 'rainDetected',
          messageKey: 'rainSensorDetectedPrecipitation'
        }
      }
    };

    let condition;
    if (value > maxOptimal) condition = 'high';
    else if (value < minOptimal) condition = 'low';
    else return null;

    const alert = alertData[sensorType]?.[condition];
    if (!alert) return null;

    return {
      titleKey: alert.titleKey,
      messageKey: alert.messageKey,
      value: value,
      sensorType: sensorType
    };
  }, []);

  // Helper function to format Drive upload timestamp
  const formatDriveUploadTime = useCallback((driveTime) => {
    if (!driveTime) return 'Unknown upload time';
    
    try {
      const date = new Date(driveTime);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      
      if (diffMins < 1) return 'Just uploaded';
      if (diffMins < 60) return `Uploaded ${diffMins}m ago`;
      if (diffHours < 24) return `Uploaded ${diffHours}h ago`;
      return `Uploaded ${date.toLocaleDateString()}`;
    } catch (error) {
      return 'Invalid upload time';
    }
  }, []);

  // 1. SENSOR ALERTS GENERATION (WARNING LEVEL) - FIXED with translation keys
  useEffect(() => {
    if (!alertPreferences.enableSensorAlerts || !sensorData) {
      setSensorAlerts([]);
      return;
    }

    const newSensorAlerts = [];
    let sensorTimestamp;
    
    try {
      if (sensorData.timestamp) {
        const timeStr = sensorData.timestamp.toString();
        if (timeStr.includes('-') && timeStr.includes(':')) {
          const [datePart, timePart] = timeStr.split(' ');
          const [day, month, year] = datePart.split('-');
          const [hour, minute, second] = timePart.split(':');
          
          sensorTimestamp = new Date(
            parseInt(year), 
            parseInt(month) - 1, 
            parseInt(day), 
            parseInt(hour), 
            parseInt(minute), 
            parseInt(second)
          ).toISOString();
        } else {
          sensorTimestamp = new Date(sensorData.timestamp).toISOString();
        }
      } else {
        sensorTimestamp = new Date().toISOString();
      }
    } catch (error) {
      console.error('Error parsing sensor timestamp:', error);
      sensorTimestamp = new Date().toISOString();
    }

    Object.keys(SENSOR_THRESHOLDS).forEach(sensorType => {
      const sensor = sensorData[sensorType];
      if (!sensor) return;

      const value = parseFloat(sensor.value);
      const threshold = SENSOR_THRESHOLDS[sensorType];
      const { optimal } = threshold;

      // Generate WARNING alerts when outside OPTIMAL range (preventive)
      let shouldAlert = false;
      if (value < optimal[0] || value > optimal[1]) {
        shouldAlert = true;
      }

      if (shouldAlert) {
        const alertData = generateSensorAlertData(sensorType, value, 'warning', optimal);
        if (alertData) {
          const alertId = `sensor_${sensorType}_${Math.floor(Date.now() / 60000)}`; // Group by minute
          
          if (!alertPreferences.dismissedAlerts.includes(alertId)) {
            newSensorAlerts.push({
              id: alertId,
              type: 'warning', // Always warning for sensors
              category: 'sensor',
              sensor: sensorType,
              titleKey: alertData.titleKey,
              messageKey: alertData.messageKey,
              value: value,
              sensorType: sensorType,
              timestamp: sensorTimestamp,
              read: alertPreferences.readAlerts.includes(alertId),
              location: userLocation,
              sourceKey: 'sensorAlert',
              // FIXED: Add navigation hint for redirect
              redirectTo: '/dashboard'
            });
          }
        }
      }
    });

    console.log('🚨 Generated sensor alerts:', newSensorAlerts.length);
    setSensorAlerts(newSensorAlerts);
  }, [sensorData, alertPreferences, userLocation, generateSensorAlertData]);

  // 2. MANUAL DETECTION ALERTS - FIXED to show disease names directly
  useEffect(() => {
    if (!alertPreferences.enableManualAlerts) {
      setManualDetectionAlerts([]);
      return;
    }

    try {
      const newManualAlerts = [];
      const manualHistory = JSON.parse(localStorage.getItem('diseaseAnalysisHistory') || '[]');
      
      // Show ALL manual detection history items as alerts (not time-limited)
      manualHistory.forEach(detection => {
        const alertId = `manual_${detection.id}`;

        // Skip dismissed alerts only
        if (!alertPreferences.dismissedAlerts.includes(alertId)) {
          newManualAlerts.push({
            id: alertId,
            type: detection.disease !== 'Healthy' ? 'critical' : 'info',
            category: 'manual',
            // FIXED: Store disease name directly instead of using titleKey
            disease: detection.disease,
            confidence: detection.confidence,
            severity: detection.severity || (detection.disease === 'Healthy' ? 'None' : 'High'),
            messageKey: 'manualAnalysisComplete',
            sourceKey: 'manualUpload',
            timestamp: detection.timestamp,
            location: userLocation,
            read: alertPreferences.readAlerts.includes(alertId),
            // Include image data for display
            hasVisualization: !!detection.hasVisualization,
            detectedRegions: detection.detectedRegions || 0,
            // FIXED: Add navigation hint
            redirectTo: '/detection',
            // FIXED: Add upload time context
            uploadTime: detection.timestamp,
            uploadContext: `Manual upload analyzed ${formatDriveUploadTime(detection.timestamp)}`
          });
        }
      });

      setManualDetectionAlerts(newManualAlerts);
      console.log('📋 Manual detection alerts:', newManualAlerts.length);
    } catch (error) {
      console.error('Failed to load manual detection alerts:', error);
      setManualDetectionAlerts([]);
    }
  }, [alertPreferences, userLocation, formatDriveUploadTime]);

  // 3. LIVE DETECTION ALERTS - FIXED to show disease names directly
  useEffect(() => {
    if (!alertPreferences.enableLiveAlerts) {
      setLiveDetectionAlerts([]);
      return;
    }

    try {
      const newLiveAlerts = [];
      const liveHistory = JSON.parse(localStorage.getItem('liveDetectionHistory') || '[]');
      
      // Show ALL live detection history items as alerts (not time-limited)
      liveHistory.forEach(detection => {
        const alertId = `live_${detection.historyId || detection.id}`;

        // Skip dismissed alerts only
        if (!alertPreferences.dismissedAlerts.includes(alertId)) {
          // FIXED: Get the original Drive upload time
          const driveUploadTime = detection.imageData?.createdTime || detection.driveUploadTime || detection.timestamp;
          
          newLiveAlerts.push({
            id: alertId,
            type: detection.detection?.disease !== 'Healthy' ? 'critical' : 'info',
            category: 'live',
            // FIXED: Store disease name directly instead of using titleKey
            disease: detection.detection.disease,
            confidence: detection.detection.confidence,
            severity: detection.detection.severity || (detection.detection.disease === 'Healthy' ? 'None' : 'High'),
            messageKey: 'liveAnalysisComplete',
            sourceKey: 'liveCamera',
            timestamp: detection.timestamp, // Analysis timestamp
            driveUploadTime: driveUploadTime, // Original image timestamp
            location: userLocation,
            read: alertPreferences.readAlerts.includes(alertId),
            // Include camera and image data
            camera: detection.camera,
            hasVisualization: !!detection.visualizationImage,
            detectedRegions: detection.detection.detectedRegions || 0,
            // FIXED: Navigate to live monitoring section
            redirectTo: '/detection?tab=live',
            // FIXED: Enhanced context with Drive upload time
            driveFileName: detection.imageData?.name || detection.driveFileName || 'Unknown file',
            uploadContext: `ESP32-CAM image ${formatDriveUploadTime(driveUploadTime)} → AI analyzed ${formatDriveUploadTime(detection.timestamp)}`
          });
        }
      });

      setLiveDetectionAlerts(newLiveAlerts);
      console.log('📹 Live detection alerts:', newLiveAlerts.length);
    } catch (error) {
      console.error('Failed to load live detection alerts:', error);
      setLiveDetectionAlerts([]);
    }
  }, [alertPreferences, userLocation, formatDriveUploadTime]);

  // Alert management functions
  const handleMarkAsRead = useCallback((alertId) => {
    setAlertPreferences(prev => ({
      ...prev,
      readAlerts: [...prev.readAlerts.filter(id => id !== alertId), alertId]
    }));
  }, []);

  const handleDismissAlert = useCallback((alertId) => {
    setAlertPreferences(prev => ({
      ...prev,
      dismissedAlerts: [...prev.dismissedAlerts.filter(id => id !== alertId), alertId]
    }));
  }, []);

  const updatePreferences = useCallback((updates) => {
    setAlertPreferences(prev => ({ ...prev, ...updates }));
  }, []);

  return {
    sensorAlerts,           // Warning level - preventive
    manualDetectionAlerts,  // All manual detection history as alerts
    liveDetectionAlerts,    // All live detection history as alerts
    alertPreferences,
    updatePreferences,
    handleMarkAsRead,
    handleDismissAlert
  };
};