// src/translations/marathi.js
export const marathiTranslations = {
  // ===== CORE NAVIGATION & BASIC =====
  dashboard: "डॅशबोर्ड",
  diseaseDetection: "रोग शोधणे",
  settings: "सेटिंग्स",
  help: "मदत",
  logout: "लॉगआउट",
  
  // ===== DASHBOARD TRANSLATIONS =====
  welcomeToGrapeGuard: "ग्रेप गार्डमध्ये आपले स्वागत",
  systemOverview: "सिस्टम विहंगावलोकन",
  environmentalStatus: "पर्यावरणीय स्थिती",
  additionalSensors: "अतिरिक्त सेन्सर",
  sensorTrends: "सेन्सर ट्रेंड (गेले 12 तास)",
  farmHealth: "शेताचे आरोग्य",
  quickActions: "त्वरित कृती",
  connectedToSensors: "सेन्सरशी जोडले",
  lastUpdated: "शेवटचे अपडेट",
  refresh: "रिफ्रेश करा",
  excellent: "उत्कृष्ट",
  diseaseRiskStatus: "रोग जोखीम स्थिती: कमी",
  noImmediateRisks: "सध्याची पर्यावरणीय परिस्थिती अनुकूल आहे। पश्चिम महाराष्ट्रातील द्राक्ष जातींसाठी कोणताही तात्काळ रोग धोका आढळला नाही।",
  
  // Sensor names
  soilMoisture: "मातीची ओलावा",
  temperature: "तापमान",
  humidity: "आर्द्रता",
  lightIntensity: "प्रकाशाची तीव्रता",
  rainSensor: "पाऊस सेन्सर",
  activeBattery: "सक्रिय बॅटरी",
  batteryVoltage: "बॅटरी व्होल्टेज",
  
  // Status
  online: "ऑनलाइन",
  offline: "ऑफलाइन",
  good: "चांगले",
  warning: "चेतावणी",
  critical: "गंभीर",
  active: "सक्रिय",
  
  // Actions
  viewAlerts: "सूचना पहा",
  downloadReport: "अहवाल डाउनलोड करा",
  
  // Chart labels
  temperatureTrend: "तापमान ट्रेंड",
  humidityTrend: "आर्द्रता ट्रेंड",
  soilMoistureTrend: "मातीची ओलावा ट्रेंड",
  lightIntensityTrend: "प्रकाश तीव्रता ट्रेंड",
  rainSensorTrend: "पाऊस सेन्सर ट्रेंड",
  
  // Loading states
  loadingSensorData: "सेन्सर डेटा लोड होत आहे...",
  loadingChartData: "चार्ट डेटा लोड होत आहे...",
  errorLoadingSensorData: "सेन्सर डेटा लोड करण्यात त्रुटी",
  noDataAvailable: "कोणताही डेटा उपलब्ध नाही",
  waitingForSensorData: "सेन्सर डेटाची प्रतीक्षा...",
  
  // Time labels
  today: "आज",
  yesterday: "काल",
  time: "वेळ",
  justNow: "आत्ताच",
  ago: "आधी",

  // ===== DETECTION PAGE TRANSLATIONS =====
  
  // Main Detection Page
  manualUpload: "मॅन्युअल अपलोड",
  liveCameraMonitoring: "लाइव्ह कॅमेरा मॉनिटरिंग",
  manualUploadDescription: "वैयक्तिक द्राक्षाच्या पानांच्या प्रतिमा मॅन्युअल अपलोड आणि विश्लेषित करा",
  liveCameraDescription: "गूगल ड्राइव्हमधून ESP32-CAM प्रतिमांचे स्वयंचलित निरीक्षण आणि रियल-टाइम रोग शोधणे",
  
  // Model Status
  detectronModel: "डिटेक्ट्रॉन2 मॉडेल",
  readyForDetection: "शोधणेसाठी तयार",
  serverConnectionFailed: "सर्व्हर कनेक्शन अयशस्वी",
  refreshModel: "रिफ्रेश करा",
  refreshingModel: "मॉडेल रिफ्रेश होत आहे...",
  processing: "प्रक्रिया होत आहे...",
  
  // Model Errors
  detectronServerError: "डिटेक्ट्रॉन2 सर्व्हर त्रुटी",
  detectronServerOffline: "डिटेक्ट्रॉन2 सर्व्हर ऑफलाइन",
  imageUploadDisabled: "सर्व्हर चालू होईपर्यंत प्रतिमा अपलोड अक्षम",
  startDetectronServer: "प्रतिमा विश्लेषण सक्षम करण्यासाठी डिटेक्ट्रॉन2 सर्व्हर सुरू करा",
  serverRequiredForAI: "AI विश्लेषणासाठी सर्व्हर आवश्यक",
  cannotAnalyze: "विश्लेषण करू शकत नाही: डिटेक्ट्रॉन2 सर्व्हर चालू नाही",
  
  // Image Upload Section
  uploadGrapeLeafImage: "द्राक्षाच्या पानाची प्रतिमा अपलोड करा",
  dragDropImage: "तुमच्या द्राक्षाच्या पानाची प्रतिमा येथे ड्रॅग आणि ड्रॉप करा",
  clickToBrowse: "किंवा फाइल ब्राउझ करण्यासाठी क्लिक करा • कमाल 5MB • JPG, PNG, WEBP समर्थित",
  selectImage: "प्रतिमा निवडा",
  chooseDifferentImage: "वेगळी प्रतिमा निवडा",
  uploadDisabled: "अपलोड अक्षम",
  
  // Analysis
  runAIAnalysis: "AI विश्लेषण चालवा",
  analyzingWithAI: "AI सह विश्लेषण...",
  aiAnalysisResults: "AI विश्लेषण परिणाम",
  uploadImageToSeeResults: "AI विश्लेषण परिणाम पाहण्यासाठी प्रतिमा अपलोड करा",
  aiIsAnalyzing: "AI तुमच्या प्रतिमेचे विश्लेषण करत आहे...",
  
  // Results Section
  aiDetectionVisualization: "AI शोध दृश्यमान:",
  aiEnhancedVisualization: "शोधलेल्या क्षेत्रांसह AI-सुधारित दृश्यमान हायलाइट केले",
  affectedRegions: "प्रभावित क्षेत्र",
  healthyArea: "निरोगी क्षेत्र",
  aiConfidence: "AI विश्वसनीयता",
  treatmentRecommendations: "उपचार शिफारसी:",
  aiAnalysisComparison: "AI विश्लेषण तुलना",
  originalImage: "मूळ प्रतिमा",
  aiDetectionResults: "AI शोध परिणाम",
  noVisualizationAvailable: "कोणतेही दृश्यमान उपलब्ध नाही",
  detectionDetails: "शोध तपशील:",
  processingTime: "प्रक्रिया वेळ",
  model: "मॉडेल",
  storage: "स्टोरेज",
  optimized: "अनुकूलित",
  downloadVisualization: "दृश्यमान डाउनलोड करा",
  close: "बंद करा",
  
  // Analysis History
  recentAnalysisHistory: "अलीकडील विश्लेषण इतिहास",
  clearHistory: "इतिहास साफ करा",
  noAnalysisHistoryYet: "अजून कोणताही विश्लेषण इतिहास नाही",
  uploadAndAnalyzeImages: "तुमचा इतिहास येथे पाहण्यासाठी प्रतिमा अपलोड आणि विश्लेषित करा",
  deleteThisAnalysis: "हे विश्लेषण हटवा",
  analysisResult: "विश्लेषण परिणाम",
  confidence: "विश्वसनीयता",
  analyzed: "विश्लेषित",
  
  // Live Camera Feed
  autoDetectThroughCamera: "कॅमेरा द्वारे ऑटो डिटेक्ट",
  clearResults: "परिणाम साफ करा",
  googleDriveConnected: "गूगल ड्राइव्ह कनेक्ट केले",
  aiModelReady: "AI मॉडेल तयार",
  aiModelFailed: "AI मॉडेल अयशस्वी",
  aiModelLoading: "AI मॉडेल लोड होत आहे...",
  noAIDetectionsYet: "अजून कोणतेही AI शोध नाहीत",
  clickAutoDetect: "डिटेक्ट्रॉन2 सह नवीनतम प्रतिमा प्रक्रिया करण्यासाठी \"कॅमेरा द्वारे ऑटो डिटेक्ट\" वर क्लिक करा",
  aiDetected: "AI द्वारे शोधले",
  regions: "क्षेत्रे",
  
  // Disease Names
  karpaAnthracnose: "कर्पा (अँथ्रकनोज)",
  bhuriPowderyMildew: "भुरी (पावडरी मिल्ड्यू)",
  bokadlelaBorer: "बोकाडलेला (बोरर संसर्ग)",
  davnyaDownyMildew: "दवयाचा (डाउनी मिल्ड्यू)",
  healthy: "निरोगी",
  
  // Analysis Status
  analysisError: "विश्लेषण त्रुटी",
  detectionError: "शोध त्रुटी",
  modelNotAvailable: "मॉडेल उपलब्ध नाही",
  
  // Camera Specific
  camera: "कॅमेरा",
  captured: "कॅप्चर केले",
  espCam: "ESP32-CAM",
  realDetectron: "वास्तविक डिटेक्ट्रॉन2",
  drive: "ड्राइव्ह",
  
  // Actions
  compare: "तुलना करा",
  download: "डाउनलोड करा",
  compareImages: "प्रतिमांची तुलना करा",
  downloadResult: "परिणाम डाउनलोड करा",
  downloadAnalysis: "विश्लेषण डाउनलोड करा",

  // Settings
  languageSettings: "भाषा सेटिंग्स",
  selectLanguage: "भाषा निवडा",
  choosePreferredLanguage: "तुमची पसंतीची भाषा निवडा। संपूर्ण वेबसाइटचे भाषांतर होईल।",
  saveSettings: "सेटिंग्स जतन करा",
  settingsSavedSuccessfully: "सेटिंग्स यशस्वीरित्या जतन झाल्या!",

  // Severity levels
  high: "उच्च",
  medium: "मध्यम",
  low: "कमी",
  none: "काहीही नाही",
  unknown: "अज्ञात",

  // Additional labels for live monitoring
  image: "प्रतिमा",
  processedIn: "मध्ये प्रक्रिया केली",
  severity: "तीव्रता",
  disease: "रोग",
  source: "स्रोत",
  waitingForAIDetection: "AI शोधाची प्रतीक्षा",
  readyForProcessing: "प्रक्रियेसाठी तयार",

  // Treatment Recommendations (Marathi)
  maintainProperIrrigation: "योग्य पाणी पुरवठा वेळापत्रक राखा",
  applyPreventiveFungicide: "मासिक प्रतिबंधात्मक बुरशीनाशक फवारणी करा",
  ensureBalancedNutrition: "संतुलित पोषण सुनिश्चित करा (NPK + सूक्ष्म पोषक द्रव्ये)",
  keepVineyardClean: "द्राक्षमळा गळालेली पाने स्वच्छ ठेवा",
  monitorWeatherConditions: "रोगाच्या प्रादुर्भावासाठी हवामान परिस्थितीचे निरीक्षण करा",
  removeInfectedLeaves: "संक्रमित पाने लगेच काढा आणि जाळा",
  sprayChlorothalonil: "क्लोरोथॅलोनिल (0.2%) किंवा कार्बेंडाझिम (0.1%) फवारणी करा",
  improveAirCirculation: "दाट पाने छाटून हवेचे संचार सुधारा",
  avoidOverheadIrrigation: "दमट हवामानात वरून पाणी देणे टाळा",
  applyCopperFungicide: "प्रतिबंधात्मक उपाय म्हणून तांबे आधारित बुरशीनाशक लावा",

  // ===== ALERTS PAGE TRANSLATIONS =====

  // Loading and basic states
  loadingAlerts: "अलर्ट लोड होत आहेत...",
  unknownTime: "अज्ञात वेळ",
  noTimestamp: "कोणताही टाइमस्टॅम्प उपलब्ध नाही",
  invalidTime: "अवैध वेळ",
  invalidTimestamp: "अवैध टाइमस्टॅम्प स्वरूप",
  parseError: "पार्स त्रुटी",
  alert: "अलर्ट",
  detected: "आढळले",
  
  // Time formatting
  minutesAgo: "मिनिटे आधी",
  hoursAgo: "तास आधी",
  
  // Alert page main elements
  criticalDiseases: "गंभीर रोग",
  sensorWarnings: "सेन्सर चेतावणी",
  unread: "न वाचलेले",
  all: "सर्व",
  criticalAlerts: "गंभीर", 
  warnings: "चेतावणी",
  
  // Alert sections
  environmentalSensors: "पर्यावरणीय सेन्सर",
  manualDetection: "मॅन्युअल शोध",
  liveCameraMonitoringAlerts: "लाइव्ह कॅमेरा मॉनिटरिंग",
  
  // Empty state messages
  allSensorsOptimal: "सर्व सेन्सर अनुकूल श्रेणीत",
  noRecentManualDetections: "कोणतेही अलीकडील मॅन्युअल शोध नाहीत",
  noRecentCameraDetections: "कोणतेही अलीकडील कॅमेरा शोध नाहीत",
  
  // Alert card actions
  viewDashboard: "डॅशबोर्ड पहा",
  goToDetection: "शोधणे वर जा",
  viewLiveFeed: "लाइव्ह फीड पहा",
  viewSource: "स्रोत पहा",
  markAsRead: "वाचले म्हणून चिन्हांकित करा",
  dismissAlert: "अलर्ट डिसमिस करा",
  
  // Alert context labels
  sensorReading: "सेन्सर रीडिंग",
  manualUploadAnalyzed: "मॅन्युअल अपलोड विश्लेषित",
  manualAnalysis: "मॅन्युअल विश्लेषण",
  aiAnalyzed: "AI द्वारा विश्लेषित",
  upload: "अपलोड",
  sensor: "सेन्सर",
  analyzedWith: "सह विश्लेषित",
  liveCamera: "लाइव्ह कॅमेरा",
  
  // Alert settings
  alertPreferences: "अलर्ट प्राधान्ये",
  alertCategories: "अलर्ट श्रेणी",
  environmentalSensorAlerts: "पर्यावरणीय सेन्सर अलर्ट",
  manualDetectionAlerts: "मॅन्युअल शोध अलर्ट",
  liveCameraAlerts: "लाइव्ह कॅमेरा अलर्ट",
  alertBehavior: "अलर्ट वर्तन",
  showCriticalAlertsOnly: "फक्त गंभीर अलर्ट दाखवा",
  autoRefreshAlerts: "अलर्ट स्वयं रिफ्रेश करा (दर 3 मिनिटांनी)",
  manageAlerts: "अलर्ट व्यवस्थापित करा",
  dismissedAlerts: "डिसमिस केलेले अलर्ट",
  readAlerts: "वाचलेले अलर्ट",
  clearDismissed: "डिसमिस केलेले साफ करा",
  markAllUnread: "सर्व अवाचित चिन्हांकित करा",
  
  // Sensor alert titles
  highTemperatureAlert: "🌡️ उच्च तापमान चेतावणी",
  lowTemperatureAlert: "🥶 कमी तापमान चेतावणी",
  highHumidityAlert: "💧 उच्च आर्द्रता चेतावणी",
  lowHumidityAlert: "🏜️ कमी आर्द्रता चेतावणी",
  excessSoilMoisture: "🌊 अतिरिक्त मातीची ओलावा",
  lowSoilMoisture: "🏜️ कमी मातीची ओलावा",
  lowBatteryVoltage: "🔋 कमी बॅटरी व्होल्टेज",
  rainDetected: "🌧️ पाऊस आढळला",
  
  // Sensor alert messages (simplified)
  temperatureExceedsOptimal: "तापमान निगरानी चेतावणी",
  temperatureBelowOptimal: "तापमान निगरानी चेतावणी",
  humidityExceedsOptimal: "आर्द्रता निगरानी चेतावणी", 
  humidityBelowOptimal: "आर्द्रता निगरानी चेतावणी",
  soilMoistureExceedsOptimal: "मातीची ओलावा निगरानी चेतावणी",
  soilMoistureBelowOptimal: "मातीची ओलावा निगरानी चेतावणी",
  batteryVoltageBelowOptimal: "बॅटरी निगरानी चेतावणी",
  rainSensorDetectedPrecipitation: "हवामान निगरानी चेतावणी",
  
  // Detection alert titles and messages
  healthyAnalysis: "निरोगी विश्लेषण",
  diseaseDetected: "रोग आढळला",
  manualAnalysisComplete: "मॅन्युअल विश्लेषण पूर्ण",
  liveAnalysisComplete: "लाइव्ह निगरानीत आढळले",
  
  // Source labels
  sensorAlert: "पर्यावरणीय सेन्सर",

  // ===== AUTHENTICATION PAGES =====
  
  // Login Page
  welcomeBack: "परत आपले स्वागत आहे! आपल्या शेती डॅशबोर्डमध्ये साइन इन करा",
  signIn: "साइन इन",
  emailAddress: "ईमेल पत्ता",
  password: "पासवर्ड",
  dontHaveAccount: "खाते नाही?",
  signUpHere: "येथे साइन अप करा",
  
  // Register Page
  joinGrapeGuard: "ग्रेप गार्डमध्ये सामील व्हा",
  createAccountDescription: "आपल्या शेताचे निरीक्षण सुरू करण्यासाठी आपले खाते तयार करा",
  fullName: "पूर्ण नाव",
  farmName: "शेताचे नाव",
  phoneNumber: "फोन नंबर",
  location: "स्थान",
  confirmPassword: "पासवर्डची पुष्टी करा",
  createAccount: "खाते तयार करा",
  alreadyHaveAccount: "आधीच खाते आहे?",
  signInHere: "येथे साइन इन करा",
  cityState: "शहर, राज्य",
  passwordsDoNotMatch: "पासवर्ड जुळत नाहीत",
  passwordTooShort: "पासवर्ड किमान 6 अक्षरांचा असावा",
  registrationFailed: "नोंदणी अयशस्वी। कृपया पुन्हा प्रयत्न करा।",

  // ===== PROFILE PAGE =====
  
  // Profile Information
  memberSince: "पासून सदस्य",
  locationNotSet: "स्थान सेट नाही",
  editProfile: "प्रोफाइल संपादित करा",
  save: "जतन करा",
  cancel: "रद्द करा",
  accountInformation: "खाते माहिती",
  emailCannotBeChanged: "ईमेल बदलला जाऊ शकत नाही",
  profileUpdatedSuccessfully: "प्रोफाइल यशस्वीरित्या अपडेट झाले!",
  failedToUpdateProfile: "प्रोफाइल अपडेट करण्यात अयशस्वी। कृपया पुन्हा प्रयत्न करा।",
  enterYourFullName: "आपले पूर्ण नाव टाका",
  enterYourFarmName: "आपल्या शेताचे नाव टाका",
  enterYourPhoneNumber: "आपला फोन नंबर टाका",
  
  // Farm Statistics
  farmStatistics: "शेत आकडेवारी",
  activeSensors: "सक्रिय सेन्सर",
  daysMonitoring: "निरीक्षण दिवस",
  imagesAnalyzed: "विश्लेषित प्रतिमा",
  systemUptime: "सिस्टम अपटाइम",

  // ===== HELP PAGE =====
  
  // Help Categories
  gettingStarted: "सुरुवात करणे",
  diseaseDetectionHelp: "रोग शोधणे",
  alertsMonitoring: "अलर्ट आणि निरीक्षण",
  technicalSupport: "तांत्रिक सहाय्य",
  questions: "प्रश्न",
  
  // FAQ Questions and Answers
  howToSetupSystem: "मी माझी ग्रेप गार्ड प्रणाली कशी सेट करू?",
  setupSystemAnswer: "ग्रेप गार्ड आपोआप आपल्या सेन्सरशी जोडते. आपले सेन्सर आपल्या द्राक्षमळ्यात योग्यरित्या बसवले आहेत आणि सिस्टमशी जोडलेले आहेत याची खात्री करा. सेन्सर सक्रिय झाल्यावर डॅशबोर्ड रियल-टाइम डेटा दाखवण्यास सुरुवात करेल.",
  
  whatSensorsNeeded: "द्राक्ष शेतीसाठी मला कोणत्या सेन्सरची गरज आहे?",
  sensorsNeededAnswer: "इष्टतम द्राक्ष निरीक्षणासाठी, आपल्याला हवे: मातीची ओलावा सेन्सर, तापमान आणि आर्द्रता सेन्सर, प्रकाश तीव्रता सेन्सर, पाऊस सेन्सर, आणि बॅटरी/पावर निरीक्षण. हे सर्व पश्चिम महाराष्ट्राच्या परिस्थितीत द्राक्षाच्या आरोग्याचे निरीक्षण करण्यास मदत करतात.",
  
  howOftenDataUpdates: "सिस्टम किती वेळा डेटा अपडेट करते?",
  dataUpdatesAnswer: "सेन्सर डेटा दर 5-10 मिनिटांनी आपोआप अपडेट होतो. आपण डॅशबोर्डवरील रिफ्रेश बटन वापरून मॅन्युअली देखील डेटा रिफ्रेश करू शकता.",
  
  aiAccuracy: "AI रोग शोधणे किती अचूक आहे?",
  aiAccuracyAnswer: "आमची AI प्रणाली विशेषतः पश्चिम महाराष्ट्रातील द्राक्ष जातींसाठी प्रशिक्षित आहे आणि पावडरी मिल्ड्यू, डाउनी मिल्ड्यू, ब्लॅक रॉट आणि अँथ्रकनोज सारख्या सामान्य रोगांचा 85-95% अचूकतेने शोध लावू शकते. अंतिम उपचार निर्णयांसाठी नेहमी कृषी तज्ञांचा सल्ला घ्या.",
  
  diseasesDetected: "कोणत्या प्रकारच्या द्राक्ष रोगांचा शोध लावला जाऊ शकतो?",
  diseasesDetectedAnswer: "सिस्टम शोधू शकते: 1) पावडरी मिल्ड्यू (भुरी रोग), 2) डाउनी मिल्ड्यू (दवयाचा रोग), 3) ब्लॅक रॉट (काळा रोग), 4) अँथ्रकनोज (कर्पा रोग). प्रत्येक महाराष्ट्र द्राक्ष जातींसाठी विशिष्ट उपचार शिफारसींसह येते.",
  
  goodPhotos: "रोग शोधणेसाठी चांगले फोटो कसे काढावेत?",
  goodPhotosAnswer: "सर्वोत्तम परिणामांसाठी: 1) चांगल्या नैसर्गिक प्रकाशात फोटो काढा, 2) लक्षणे दाखवणाऱ्या वैयक्तिक पानांवर लक्ष केंद्रित करा, 3) पान फ्रेमचा बहुतेक भाग भरत असल्याची खात्री करा, 4) अस्पष्ट किंवा गडद प्रतिमा टाळा, 5) रोग अस्पष्ट असल्यास अनेक कोनातून घ्या.",
  
  treatmentSafety: "उपचार शिफारसी माझ्या द्राक्षांसाठी सुरक्षित आहेत का?",
  treatmentSafetyAnswer: "होय, सर्व शिफारसी पश्चिम महाराष्ट्रातील द्राक्ष शेतीसाठी सेंद्रिय आणि मान्यताप्राप्त पद्धतींवर आधारित आहेत. तथापि, नेहमी प्रथम लहान क्षेत्रावर चाचणी करा आणि स्थानिक कृषी मार्गदर्शक तत्त्वांचे पालन करा.",
  
  alertColors: "विविध अलर्ट रंगांचा अर्थ काय आहे?",
  alertColorsAnswer: "लाल (गंभीर): तात्काळ कृती आवश्यक - द्राक्षाच्या आरोग्यावर परिणाम होऊ शकतो. पिवळा (चेतावणी): बारकाईने निरीक्षण करा - समस्याग्रस्त पातळीकडे कल. निळा (माहिती): सामान्य माहिती किंवा इष्टतमपासून किरकोळ विचलन.",
  
  humidityLevels: "आर्द्रतेच्या पातळीबद्दल मला कधी चिंता करावी?",
  humidityLevelsAnswer: "पश्चिम महाराष्ट्रातील द्राक्षांसाठी: आर्द्रता >85% रोगाचा धोका वाढवते (विशेषतः पावसाळ्यात), आर्द्रता <40% वनस्पतींचा ताणतणाव होऊ शकतो. निरोगी द्राक्ष वाढीसाठी इष्टतम श्रेणी 40-60% आहे.",
  
  soilMoistureLevels: "द्राक्षांसाठी इष्टतम मातीची ओलावा पातळी काय आहे?",
  soilMoistureLevelsAnswer: "द्राक्षे चांगल्या निचऱ्याची माती पसंत करतात: 30-75% ओलावा इष्टतम आहे. >85% मुळांचा सडत रोग होऊ शकतो, <20% दुष्काळी ताणतणाव निर्माण करतो. मातीच्या ओलावा वाचन आणि हवामान परिस्थितीच्या आधारे सिंचन समायोजित करा.",
  
  batteryImportance: "माझे बॅटरी व्होल्टेज का महत्वाचे आहे?",
  batteryImportanceAnswer: "9V पेक्षा कमी बॅटरी व्होल्टेज सेन्सर अपयशास कारणीभूत ठरू शकते. आवश्यकतेनुसार सिस्टम आपोआप बॅकअप बॅटरीवर स्विच करते. नियमित निरीक्षण आपल्या शेतासाठी सतत डेटा संकलन सुनिश्चित करते.",
  
  sensorsNotWorking: "माझे सेन्सर काम करणे थांबवल्यास काय करावे?",
  sensorsNotWorkingAnswer: "तपासा: 1) पावर कनेक्शन आणि बॅटरी व्होल्टेज, 2) सेन्सर भौतिक स्थिती, 3) हवामान नुकसान, 4) नेटवर्क कनेक्टिव्हिटी. समस्या कायम राहिल्यास, आपल्या सेन्सर मॉडेल आणि त्रुटी तपशीलांसह सहाय्याशी संपर्क साधा.",
  
  updateLocation: "मी माझ्या शेताचे स्थान कसे अपडेट करू?",
  updateLocationAnswer: "प्रोफाइल सेटिंग्स → खाते माहिती → स्थान फील्डवर जा. महाराष्ट्रातील आपल्या विशिष्ट क्षेत्रासाठी अचूक हवामान आणि रोग जोखीम अंदाज मिळविण्यासाठी आपले स्थान अपडेट करा.",
  
  exportData: "मी माझ्या शेताचा डेटा निर्यात करू शकतो का?",
  exportDataAnswer: "सध्या, आपण चार्ट आणि अलर्टमध्ये ऐतिहासिक डेटा पाहू शकता. डेटा निर्यात वैशिष्ट्ये लवकरच येत आहेत. आपला विश्लेषण इतिहास आपल्या संदर्भासाठी आपोआप जतन केला जातो.",
  
  dataSecure: "माझ्या शेताचा डेटा सुरक्षित आहे का?",
  dataSecureAnswer: "होय, सर्व डेटा एन्क्रिप्ट केलेला आहे आणि सुरक्षितपणे संग्रहीत आहे. फक्त आपल्याकडे आपल्या शेताच्या डेटामध्ये प्रवेश आहे. आम्ही कठोर गोपनीयता मार्गदर्शक तत्त्वांचे पालन करतो आणि आपल्या संमतीशिवाय कधीही आपली माहिती सामायिक करत नाही.",
  
  // Help Tags
  setup: "सेटअप",
  sensors: "सेन्सर",
  hardware: "हार्डवेअर",
  data: "डेटा",
  updates: "अपडेट",
  ai: "AI",
  accuracy: "अचूकता",
  diseases: "रोग",
  detection: "शोध",
  photography: "फोटोग्राफी",
  tips: "टिप्स",
  treatment: "उपचार",
  safety: "सुरक्षितता",
  alerts: "अलर्ट",
  colors: "रंग",
  thresholds: "मर्यादा",
  soil: "माती",
  irrigation: "सिंचन",
  battery: "बॅटरी",
  power: "पावर",
  troubleshooting: "समस्या निवारण",
  profile: "प्रोफाइल",
  export: "निर्यात",
  privacy: "गोपनीयता",
  security: "सुरक्षा",
  
  // Emergency Guidelines
  emergencyPlantHealth: "आपातकालीन वनस्पती आरोग्य मार्गदर्शक तत्त्वे",
  immediateActionRequired: "🚨 तात्काळ कृती आवश्यक जेव्हा:",
  callAgriculturalExpert: "📞 कृषी तज्ञाला कधी कॉल करावे:",
  temperatureHigh: "तापमान >40°C 2 तासांपेक्षा जास्त",
  soilMoistureLow: "वाढीच्या हंगामात मातीची ओलावा <15%",
  humidityHigh: "आर्द्रता >90% तापमान >25°C सह",
  multipleDiseases: "एकाच क्षेत्रात अनेक रोग अलर्ट",
  sensorFailures: "संपूर्ण शेतात अचानक सेन्सर अपयश",
  highConfidenceDisease: "उच्च तीव्रतेसह रोगाचा विश्वास >90%",
  multipleSimultaneous: "एकाच वेळी अनेक प्रकारचे रोग आढळले",
  unusualSymptoms: "AI अंदाजांशी जुळत नसलेली असामान्य लक्षणे",
  widespreadStress: "संपूर्ण द्राक्षमळ्यात व्यापक वनस्पती ताणतणाव",
  beforeChemicals: "कोणतेही रासायनिक उपचार लागू करण्यापूर्वी",
  
  // Version Info
  grapeGuardVersion: "ग्रेप गार्ड आवृत्ती:",
  region: "प्रदेश:",
  crops: "पिके:",
  westernMaharashtra: "पश्चिम महाराष्ट्र",
  grapeVarieties: "द्राक्ष जाती",
};