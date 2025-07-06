// src/translations/hindi.js
export const hindiTranslations = {
  // ===== CORE NAVIGATION & BASIC =====
  dashboard: "डैशबोर्ड",
  diseaseDetection: "रोग का पता लगाना",
  settings: "सेटिंग्स",
  help: "मदद",
  logout: "लॉगआउट",
  
  // ===== DASHBOARD TRANSLATIONS =====
  welcomeToGrapeGuard: "ग्रेप गार्ड में आपका स्वागत है",
  systemOverview: "सिस्टम अवलोकन",
  environmentalStatus: "पर्यावरणीय स्थिति",
  additionalSensors: "अतिरिक्त सेंसर",
  sensorTrends: "सेंसर रुझान (पिछले 12 घंटे)",
  farmHealth: "खेत का स्वास्थ्य",
  quickActions: "त्वरित कार्य",
  connectedToSensors: "सेंसर से जुड़ा हुआ",
  lastUpdated: "अंतिम अपडेट",
  refresh: "रिफ्रेश करें",
  excellent: "उत्कृष्ट",
  diseaseRiskStatus: "रोग जोखिम स्थिति: कम",
  noImmediateRisks: "वर्तमान पर्यावरणीय स्थितियां अनुकूल हैं। पश्चिमी महाराष्ट्र अंगूर किस्मों के लिए कोई तत्काल रोग जोखिम नहीं मिला।",
  
  // Sensor names
  soilMoisture: "मिट्टी की नमी",
  temperature: "तापमान",
  humidity: "नमी",
  lightIntensity: "प्रकाश की तीव्रता",
  rainSensor: "बारिश सेंसर",
  activeBattery: "सक्रिय बैटरी",
  batteryVoltage: "बैटरी वोल्टेज",
  
  // Status
  online: "ऑनलाइन",
  offline: "ऑफलाइन",
  good: "अच्छा",
  warning: "चेतावनी",
  critical: "गंभीर",
  active: "सक्रिय",
  
  // Actions
  viewAlerts: "अलर्ट देखें",
  downloadReport: "रिपोर्ट डाउनलोड करें",
  
  // Chart labels
  temperatureTrend: "तापमान रुझान",
  humidityTrend: "नमी रुझान",
  soilMoistureTrend: "मिट्टी की नमी रुझान",
  lightIntensityTrend: "प्रकाश तीव्रता रुझान",
  rainSensorTrend: "बारिश सेंसर रुझान",
  
  // Loading states
  loadingSensorData: "सेंसर डेटा लोड हो रहा है...",
  loadingChartData: "चार्ट डेटा लोड हो रहा है...",
  errorLoadingSensorData: "सेंसर डेटा लोड करने में त्रुटि",
  noDataAvailable: "कोई डेटा उपलब्ध नहीं",
  waitingForSensorData: "सेंसर डेटा का इंतजार...",
  
  // Time labels
  today: "आज",
  yesterday: "कल",
  time: "समय",
  justNow: "अभी",
  ago: "पहले",

  // ===== DETECTION PAGE TRANSLATIONS =====
  
  // Main Detection Page
  manualUpload: "मैन्युअल अपलोड",
  liveCameraMonitoring: "लाइव कैमरा मॉनिटरिंग",
  manualUploadDescription: "व्यक्तिगत अंगूर के पत्तों की तस्वीरों को मैन्युअल रूप से अपलोड और विश्लेषित करें",
  liveCameraDescription: "गूगल ड्राइव से ESP32-CAM छवियों की स्वचालित निगरानी के साथ रियल-टाइम रोग का पता लगाना",
  
  // Model Status
  detectronModel: "डिटेक्ट्रॉन2 मॉडल",
  readyForDetection: "पहचान के लिए तैयार",
  serverConnectionFailed: "सर्वर कनेक्शन विफल",
  refreshModel: "रिफ्रेश करें",
  refreshingModel: "मॉडल रिफ्रेश हो रहा है...",
  processing: "प्रोसेसिंग...",
  
  // Model Errors
  detectronServerError: "डिटेक्ट्रॉन2 सर्वर त्रुटि",
  detectronServerOffline: "डिटेक्ट्रॉन2 सर्वर ऑफलाइन",
  imageUploadDisabled: "सर्वर चलने तक छवि अपलोड अक्षम",
  startDetectronServer: "छवि विश्लेषण सक्षम करने के लिए डिटेक्ट्रॉन2 सर्वर शुरू करें",
  serverRequiredForAI: "AI विश्लेषण के लिए सर्वर आवश्यक",
  cannotAnalyze: "विश्लेषण नहीं कर सकते: डिटेक्ट्रॉन2 सर्वर नहीं चल रहा",
  
  // Image Upload Section
  uploadGrapeLeafImage: "अंगूर के पत्ते की छवि अपलोड करें",
  dragDropImage: "अपनी अंगूर के पत्ते की छवि यहाँ ड्रैग और ड्रॉप करें",
  clickToBrowse: "या फ़ाइल ब्राउज़ करने के लिए क्लिक करें • अधिकतम 5MB • JPG, PNG, WEBP समर्थित",
  selectImage: "छवि चुनें",
  chooseDifferentImage: "अलग छवि चुनें",
  uploadDisabled: "अपलोड अक्षम",
  
  // Analysis
  runAIAnalysis: "AI विश्लेषण चलाएं",
  analyzingWithAI: "AI के साथ विश्लेषण...",
  aiAnalysisResults: "AI विश्लेषण परिणाम",
  uploadImageToSeeResults: "AI विश्लेषण परिणाम देखने के लिए एक छवि अपलोड करें",
  aiIsAnalyzing: "AI आपकी छवि का विश्लेषण कर रहा है...",
  
  // Results Section
  aiDetectionVisualization: "AI पहचान दृश्यीकरण:",
  aiEnhancedVisualization: "पहचाने गए क्षेत्रों के साथ AI-संवर्धित दृश्यीकरण हाइलाइट किया गया",
  affectedRegions: "प्रभावित क्षेत्र",
  healthyArea: "स्वस्थ क्षेत्र",
  aiConfidence: "AI विश्वसनीयता",
  treatmentRecommendations: "उपचार सिफारिशें:",
  aiAnalysisComparison: "AI विश्लेषण तुलना",
  originalImage: "मूल छवि",
  aiDetectionResults: "AI पहचान परिणाम",
  noVisualizationAvailable: "कोई दृश्यीकरण उपलब्ध नहीं",
  detectionDetails: "पहचान विवरण:",
  processingTime: "प्रोसेसिंग समय",
  model: "मॉडल",
  storage: "स्टोरेज",
  optimized: "अनुकूलित",
  downloadVisualization: "दृश्यीकरण डाउनलोड करें",
  close: "बंद करें",
  
  // Analysis History
  recentAnalysisHistory: "हाल का विश्लेषण इतिहास",
  clearHistory: "इतिहास साफ़ करें",
  noAnalysisHistoryYet: "अभी तक कोई विश्लेषण इतिहास नहीं",
  uploadAndAnalyzeImages: "अपना इतिहास यहाँ देखने के लिए छवियाँ अपलोड और विश्लेषित करें",
  deleteThisAnalysis: "इस विश्लेषण को हटाएं",
  analysisResult: "विश्लेषण परिणाम",
  confidence: "विश्वसनीयता",
  analyzed: "विश्लेषित",
  
  // Live Camera Feed
  autoDetectThroughCamera: "कैमरा के माध्यम से ऑटो डिटेक्ट",
  clearResults: "परिणाम साफ़ करें",
  googleDriveConnected: "गूगल ड्राइव कनेक्टेड",
  aiModelReady: "AI मॉडल तैयार",
  aiModelFailed: "AI मॉडल विफल",
  aiModelLoading: "AI मॉडल लोड हो रहा है...",
  noAIDetectionsYet: "अभी तक कोई AI पहचान नहीं",
  clickAutoDetect: "डिटेक्ट्रॉन2 के साथ नवीनतम छवियों को प्रोसेस करने के लिए \"कैमरा के माध्यम से ऑटो डिटेक्ट\" पर क्लिक करें",
  aiDetected: "AI द्वारा पहचाना गया",
  regions: "क्षेत्र",
  
  // Disease Names
  karpaAnthracnose: "कर्पा (एंथ्रैक्नोस)",
  bhuriPowderyMildew: "भुरी (पाउडरी मिल्ड्यू)",
  bokadlelaBorer: "बोकाडलेला (बोरर संक्रमण)",
  davnyaDownyMildew: "दवयाचा (डाउनी मिल्ड्यू)",
  healthy: "स्वस्थ",
  
  // Analysis Status
  analysisError: "विश्लेषण त्रुटि",
  detectionError: "पहचान त्रुटि",
  modelNotAvailable: "मॉडल उपलब्ध नहीं",
  
  // Camera Specific
  camera: "कैमरा",
  captured: "कैप्चर किया गया",
  espCam: "ESP32-CAM",
  realDetectron: "वास्तविक डिटेक्ट्रॉन2",
  drive: "ड्राइव",
  
  // Actions
  compare: "तुलना करें",
  download: "डाउनलोड करें",
  compareImages: "छवियों की तुलना करें",
  downloadResult: "परिणाम डाउनलोड करें",
  downloadAnalysis: "विश्लेषण डाउनलोड करें",

  // Settings
  languageSettings: "भाषा सेटिंग्स",
  selectLanguage: "भाषा चुनें",
  choosePreferredLanguage: "अपनी पसंदीदा भाषा चुनें। पूरी वेबसाइट का अनुवाद हो जाएगा।",
  saveSettings: "सेटिंग्स सेव करें",
  settingsSavedSuccessfully: "सेटिंग्स सफलतापूर्वक सेव हो गईं!",

  // Severity levels
  high: "उच्च",
  medium: "मध्यम",
  low: "कम",
  none: "कोई नहीं",

  // Additional labels for live monitoring
  image: "छवि",
  processedIn: "में प्रोसेस किया गया",
  severity: "गंभीरता",
  disease: "रोग",
  source: "स्रोत",
  waitingForAIDetection: "AI पहचान की प्रतीक्षा",
  readyForProcessing: "प्रोसेसिंग के लिए तैयार",

  // ===== COMPLETE ALERTS PAGE TRANSLATIONS =====
  // Loading and basic states
  loadingAlerts: "अलर्ट लोड हो रहे हैं...",
  unknownTime: "अज्ञात समय",
  noTimestamp: "कोई टाइमस्टैम्प उपलब्ध नहीं",
  invalidTime: "अमान्य समय",
  invalidTimestamp: "अमान्य टाइमस्टैम्प प्रारूप",
  parseError: "पार्स त्रुटि",
  unknown: "अज्ञात",
  alert: "अलर्ट",
  detected: "पाया गया",
  
  // Time formatting
  minutesAgo: "मिनट पहले",
  hoursAgo: "घंटे पहले",
  
  // Alert page main elements
  criticalDiseases: "गंभीर रोग",
  sensorWarnings: "सेंसर चेतावनी",
  unread: "अपठित",
  all: "सभी",
  criticalAlerts: "गंभीर",
  warnings: "चेतावनी",
  
  // Alert sections
  environmentalSensors: "पर्यावरणीय सेंसर",
  manualDetection: "मैन्युअल पहचान",
  liveCameraMonitoringAlerts: "लाइव कैमरा मॉनिटरिंग",
  
  // Empty state messages
  allSensorsOptimal: "सभी सेंसर अनुकूल सीमा में",
  noRecentManualDetections: "कोई हाल की मैन्युअल पहचान नहीं",
  noRecentCameraDetections: "कोई हाल की कैमरा पहचान नहीं",
  
  // Alert card actions
  viewDashboard: "डैशबोर्ड देखें",
  goToDetection: "पहचान पर जाएं",
  viewLiveFeed: "लाइव फीड देखें",
  viewSource: "स्रोत देखें",
  markAsRead: "पढ़ा हुआ चिह्नित करें",
  dismissAlert: "अलर्ट खारिज करें",
  
  // Alert context labels
  sensorReading: "सेंसर रीडिंग",
  manualUploadAnalyzed: "मैन्युअल अपलोड विश्लेषित",
  manualAnalysis: "मैन्युअल विश्लेषण",
  aiAnalyzed: "AI द्वारा विश्लेषित",
  upload: "अपलोड",
  sensor: "सेंसर",
  analyzedWith: "के साथ विश्लेषित",
  liveCamera: "लाइव कैमरा",
  
  // Alert settings
  alertPreferences: "अलर्ट प्राथमिकताएं",
  alertCategories: "अलर्ट श्रेणियां",
  environmentalSensorAlerts: "पर्यावरणीय सेंसर अलर्ट",
  manualDetectionAlerts: "मैन्युअल पहचान अलर्ट", 
  liveCameraAlerts: "लाइव कैमरा अलर्ट",
  alertBehavior: "अलर्ट व्यवहार",
  showCriticalAlertsOnly: "केवल गंभीर अलर्ट दिखाएं",
  autoRefreshAlerts: "अलर्ट स्वतः रिफ्रेश करें (हर 3 मिनट)",
  manageAlerts: "अलर्ट प्रबंधित करें",
  dismissedAlerts: "खारिज किए गए अलर्ट",
  readAlerts: "पढ़े गए अलर्ट",
  clearDismissed: "खारिज किए गए साफ़ करें",
  markAllUnread: "सभी को अपठित चिह्नित करें",
  
  // Sensor alert titles
  highTemperatureAlert: "🌡️ उच्च तापमान चेतावनी",
  lowTemperatureAlert: "🥶 कम तापमान चेतावनी",
  highHumidityAlert: "💧 उच्च नमी चेतावनी",
  lowHumidityAlert: "🏜️ कम नमी चेतावनी",
  excessSoilMoisture: "🌊 अतिरिक्त मिट्टी की नमी",
  lowSoilMoisture: "🏜️ कम मिट्टी की नमी",
  lowBatteryVoltage: "🔋 कम बैटरी वोल्टेज",
  rainDetected: "🌧️ बारिश का पता चला",
  
  // Sensor alert messages (simplified)
  temperatureExceedsOptimal: "तापमान निगरानी चेतावनी",
  temperatureBelowOptimal: "तापमान निगरानी चेतावनी",
  humidityExceedsOptimal: "नमी निगरानी चेतावनी", 
  humidityBelowOptimal: "नमी निगरानी चेतावनी",
  soilMoistureExceedsOptimal: "मिट्टी की नमी निगरानी चेतावनी",
  soilMoistureBelowOptimal: "मिट्टी की नमी निगरानी चेतावनी",
  batteryVoltageBelowOptimal: "बैटरी निगरानी चेतावनी",
  rainSensorDetectedPrecipitation: "मौसम निगरानी चेतावनी",
  
  // Detection alert titles and messages
  healthyAnalysis: "स्वस्थ विश्लेषण",
  diseaseDetected: "रोग का पता चला",
  manualAnalysisComplete: "मैन्युअल विश्लेषण पूर्ण",
  liveAnalysisComplete: "लाइव निगरानी में पाया गया",
  
  // Source labels
  sensorAlert: "पर्यावरणीय सेंसर",

  // ===== AUTHENTICATION PAGES =====
  
  // Login Page
  welcomeBack: "वापसी पर स्वागत है! अपने कृषि डैशबोर्ड में साइन इन करें",
  signIn: "साइन इन",
  emailAddress: "ईमेल पता",
  password: "पासवर्ड",
  dontHaveAccount: "कोई खाता नहीं है?",
  signUpHere: "यहाँ साइन अप करें",
  
  // Register Page
  joinGrapeGuard: "ग्रेप गार्ड में शामिल हों",
  createAccountDescription: "अपने खेत की निगरानी शुरू करने के लिए अपना खाता बनाएं",
  fullName: "पूरा नाम",
  farmName: "खेत का नाम",
  phoneNumber: "फोन नंबर",
  location: "स्थान",
  confirmPassword: "पासवर्ड की पुष्टि करें",
  createAccount: "खाता बनाएं",
  alreadyHaveAccount: "पहले से खाता है?",
  signInHere: "यहाँ साइन इन करें",
  cityState: "शहर, राज्य",
  passwordsDoNotMatch: "पासवर्ड मेल नहीं खाते",
  passwordTooShort: "पासवर्ड कम से कम 6 अक्षर का होना चाहिए",
  registrationFailed: "पंजीकरण विफल। कृपया पुनः प्रयास करें।",

  // ===== PROFILE PAGE =====
  
  // Profile Information
  memberSince: "से सदस्य",
  locationNotSet: "स्थान सेट नहीं",
  editProfile: "प्रोफाइल संपादित करें",
  save: "सेव करें",
  cancel: "रद्द करें",
  accountInformation: "खाता जानकारी",
  emailCannotBeChanged: "ईमेल बदला नहीं जा सकता",
  profileUpdatedSuccessfully: "प्रोफाइल सफलतापूर्वक अपडेट हो गया!",
  failedToUpdateProfile: "प्रोफाइल अपडेट करने में विफल। कृपया पुनः प्रयास करें।",
  enterYourFullName: "अपना पूरा नाम दर्ज करें",
  enterYourFarmName: "अपने खेत का नाम दर्ज करें",
  enterYourPhoneNumber: "अपना फोन नंबर दर्ज करें",
  
  // Farm Statistics
  farmStatistics: "खेत के आंकड़े",
  activeSensors: "सक्रिय सेंसर",
  daysMonitoring: "निगरानी के दिन",
  imagesAnalyzed: "विश्लेषित छवियाँ",
  systemUptime: "सिस्टम अपटाइम",

  // ===== HELP PAGE =====
  
  // Help Categories
  gettingStarted: "शुरुआत करना",
  diseaseDetectionHelp: "रोग का पता लगाना",
  alertsMonitoring: "अलर्ट और निगरानी",
  technicalSupport: "तकनीकी सहायता",
  questions: "प्रश्न",
  
  // FAQ Questions and Answers
  howToSetupSystem: "मैं अपना ग्रेप गार्ड सिस्टम कैसे सेट करूं?",
  setupSystemAnswer: "ग्रेप गार्ड स्वचालित रूप से आपके सेंसर से जुड़ता है। सुनिश्चित करें कि आपके सेंसर आपके दाख की बारी में ठीक से स्थापित हैं और सिस्टम से जुड़े हैं। सेंसर सक्रिय होने पर डैशबोर्ड रियल-टाइम डेटा दिखाना शुरू कर देगा।",
  
  whatSensorsNeeded: "अंगूर की खेती के लिए मुझे किन सेंसर की आवश्यकता है?",
  sensorsNeededAnswer: "इष्टतम अंगूर निगरानी के लिए, आपको चाहिए: मिट्टी की नमी सेंसर, तापमान और आर्द्रता सेंसर, प्रकाश तीव्रता सेंसर, बारिश सेंसर, और बैटरी/पावर निगरानी। ये सभी पश्चिमी महाराष्ट्र की स्थितियों में अंगूर के स्वास्थ्य की निगरानी में मदद करते हैं।",
  
  howOftenDataUpdates: "सिस्टम कितनी बार डेटा अपडेट करता है?",
  dataUpdatesAnswer: "सेंसर डेटा स्वचालित रूप से हर 5-10 मिनट में अपडेट होता है। आप डैशबोर्ड पर रिफ्रेश बटन का उपयोग करके मैन्युअल रूप से भी डेटा रिफ्रेश कर सकते हैं।",
  
  aiAccuracy: "AI रोग का पता लगाना कितना सटीक है?",
  aiAccuracyAnswer: "हमारा AI सिस्टम विशेष रूप से पश्चिमी महाराष्ट्र अंगूर किस्मों के लिए प्रशिक्षित है और पाउडरी मिल्ड्यू, डाउनी मिल्ड्यू, ब्लैक रॉट, और एंथ्रेक्नोस जैसी सामान्य बीमारियों का 85-95% सटीकता के साथ पता लगा सकता है। अंतिम उपचार निर्णयों के लिए हमेशा कृषि विशेषज्ञों से सलाह लें।",
  
  diseasesDetected: "किस प्रकार की अंगूर की बीमारियों का पता लगाया जा सकता है?",
  diseasesDetectedAnswer: "सिस्टम का पता लगा सकता है: 1) पाउडरी मिल्ड्यू (भुरी रोग), 2) डाउनी मिल्ड्यू (दवयाचा रोग), 3) ब्लैक रॉट (काळा रोग), 4) एंथ्रेक्नोस (कर्पा रोग)। प्रत्येक महाराष्ट्र अंगूर किस्मों के लिए विशिष्ट उपचार सिफारिशों के साथ आता है।",
  
  goodPhotos: "रोग का पता लगाने के लिए अच्छी तस्वीरें कैसे लूं?",
  goodPhotosAnswer: "सर्वोत्तम परिणामों के लिए: 1) अच्छी प्राकृतिक रोशनी में तस्वीरें लें, 2) लक्षण दिखाने वाले व्यक्तिगत पत्तों पर ध्यान दें, 3) सुनिश्चित करें कि पत्ता फ्रेम का अधिकांश भाग भरता है, 4) धुंधली या अंधेरी छवियों से बचें, 5) यदि बीमारी अस्पष्ट है तो कई कोणों से लें।",
  
  treatmentSafety: "क्या उपचार सिफारिशें मेरे अंगूर के लिए सुरक्षित हैं?",
  treatmentSafetyAnswer: "हां, सभी सिफारिशें पश्चिमी महाराष्ट्र में अंगूर की खेती के लिए जैविक और अनुमोदित तरीकों पर आधारित हैं। हालांकि, हमेशा पहले एक छोटे क्षेत्र पर परीक्षण करें और स्थानीय कृषि दिशानिर्देशों का पालन करें।",
  
  alertColors: "विभिन्न अलर्ट रंगों का क्या मतलब है?",
  alertColorsAnswer: "लाल (गंभीर): तत्काल कार्रवाई आवश्यक - अंगूर के स्वास्थ्य को प्रभावित कर सकता है। पीला (चेतावनी): बारीकी से निगरानी करें - समस्याग्रस्त स्तरों की ओर बढ़ रहा है। नीला (जानकारी): सामान्य जानकारी या इष्टतम से मामूली विचलन।",
  
  humidityLevels: "आर्द्रता के स्तर के बारे में मुझे कब चिंतित होना चाहिए?",
  humidityLevelsAnswer: "पश्चिमी महाराष्ट्र में अंगूर के लिए: आर्द्रता >85% बीमारी का जोखिम बढ़ाती है (विशेष रूप से मानसून के दौरान), आर्द्रता <40% पौधे के तनाव का कारण हो सकती है। स्वस्थ अंगूर की वृद्धि के लिए इष्टतम सीमा 40-60% है।",
  
  soilMoistureLevels: "अंगूर के लिए इष्टतम मिट्टी की नमी का स्तर क्या है?",
  soilMoistureLevelsAnswer: "अंगूर अच्छी जल निकासी वाली मिट्टी पसंद करते हैं: 30-75% नमी इष्टतम है। >85% जड़ सड़न का कारण हो सकता है, <20% सूखे का तनाव पैदा करता है। मिट्टी की नमी रीडिंग और मौसम की स्थिति के आधार पर सिंचाई को समायोजित करें।",
  
  batteryImportance: "मेरा बैटरी वोल्टेज क्यों महत्वपूर्ण है?",
  batteryImportanceAnswer: "9V से कम बैटरी वोल्टेज सेंसर विफलताओं का कारण हो सकता है। जरूरत पड़ने पर सिस्टम स्वचालित रूप से बैकअप बैटरी पर स्विच हो जाता है। नियमित निगरानी आपके खेत के लिए निरंतर डेटा संग्रह सुनिश्चित करती है।",
  
  sensorsNotWorking: "यदि मेरे सेंसर काम करना बंद कर दें तो क्या करें?",
  sensorsNotWorkingAnswer: "जांचें: 1) पावर कनेक्शन और बैटरी वोल्टेज, 2) सेंसर की भौतिक स्थिति, 3) मौसम की क्षति, 4) नेटवर्क कनेक्टिविटी। यदि समस्याएं बनी रहती हैं, तो अपने सेंसर मॉडल और त्रुटि विवरण के साथ सहायता से संपर्क करें।",
  
  updateLocation: "मैं अपने खेत का स्थान कैसे अपडेट करूं?",
  updateLocationAnswer: "प्रोफाइल सेटिंग्स → खाता जानकारी → स्थान फील्ड पर जाएं। महाराष्ट्र में अपने विशिष्ट क्षेत्र के लिए सटीक मौसम और बीमारी जोखिम भविष्यवाणियां प्राप्त करने के लिए अपना स्थान अपडेट करें।",
  
  exportData: "क्या मैं अपने खेत का डेटा निर्यात कर सकता हूं?",
  exportDataAnswer: "वर्तमान में, आप चार्ट और अलर्ट में ऐतिहासिक डेटा देख सकते हैं। डेटा निर्यात सुविधाएं जल्द ही आ रही हैं। आपका विश्लेषण इतिहास आपके संदर्भ के लिए स्वचालित रूप से सहेजा जाता है।",
  
  dataSecure: "क्या मेरे खेत का डेटा सुरक्षित है?",
  dataSecureAnswer: "हां, सभी डेटा एन्क्रिप्ट किया गया है और सुरक्षित रूप से संग्रहीत है। केवल आपके पास अपने खेत के डेटा तक पहुंच है। हम सख्त गोपनीयता दिशानिर्देशों का पालन करते हैं और आपकी सहमति के बिना कभी भी आपकी जानकारी साझा नहीं करते।",
  
  // Help Tags
  setup: "सेटअप",
  sensors: "सेंसर",
  hardware: "हार्डवेयर",
  data: "डेटा",
  updates: "अपडेट",
  ai: "AI",
  accuracy: "सटीकता",
  diseases: "बीमारियां",
  detection: "पहचान",
  photography: "फोटोग्राफी",
  tips: "सुझाव",
  treatment: "उपचार",
  safety: "सुरक्षा",
  alerts: "अलर्ट",
  colors: "रंग",
  thresholds: "सीमा",
  soil: "मिट्टी",
  irrigation: "सिंचाई",
  battery: "बैटरी",
  power: "पावर",
  troubleshooting: "समस्या निवारण",
  profile: "प्रोफाइल",
  export: "निर्यात",
  privacy: "गोपनीयता",
  security: "सुरक्षा",
  
  // Emergency Guidelines
  emergencyPlantHealth: "आपातकालीन पौधे स्वास्थ्य दिशानिर्देश",
  immediateActionRequired: "🚨 तत्काल कार्रवाई आवश्यक जब:",
  callAgriculturalExpert: "📞 कृषि विशेषज्ञ को कब कॉल करें:",
  temperatureHigh: "तापमान >40°C 2 घंटे से अधिक समय तक",
  soilMoistureLow: "बढ़ते मौसम के दौरान मिट्टी की नमी <15%",
  humidityHigh: "आर्द्रता >90% तापमान >25°C के साथ",
  multipleDiseases: "एक ही क्षेत्र में कई बीमारी अलर्ट",
  sensorFailures: "पूरे खेत में अचानक सेंसर विफलता",
  highConfidenceDisease: "उच्च गंभीरता के साथ बीमारी का विश्वास >90%",
  multipleSimultaneous: "एक साथ कई प्रकार की बीमारियों का पता चला",
  unusualSymptoms: "AI भविष्यवाणियों से मेल न खाने वाले असामान्य लक्षण",
  widespreadStress: "पूरे दाख की बारी में व्यापक पौधे का तनाव",
  beforeChemicals: "कोई भी रासायनिक उपचार लागू करने से पहले",
  
  // Version Info
  grapeGuardVersion: "ग्रेप गार्ड संस्करण:",
  region: "क्षेत्र:",
  crops: "फसलें:",
  westernMaharashtra: "पश्चिमी महाराष्ट्र",
  grapeVarieties: "अंगूर किस्में",
};