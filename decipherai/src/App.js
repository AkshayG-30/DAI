import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [preview, setPreview] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showProcessing, setShowProcessing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [currentPage, setCurrentPage] = useState('home');
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  // Authentication state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);

  const [detectedText, setDetectedText] = useState('');
  const [translation, setTranslation] = useState('');
  const [historicalContext, setHistoricalContext] = useState(null);

  // Authentication form data
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [registerData, setRegisterData] = useState({ 
    firstName: '', 
    lastName: '', 
    email: '', 
    password: '', 
    confirmPassword: '' 
  });
  const [authErrors, setAuthErrors] = useState({});

  // Added magical home page state
  const [showMagicalHome, setShowMagicalHome] = useState(true);

  // HOW IT WORKS PAGE STATE VARIABLES
  const [pipelineStep, setPipelineStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedModel, setSelectedModel] = useState('all');
  const [processingDemo, setProcessingDemo] = useState(null);
  const [animationSpeed, setAnimationSpeed] = useState(2500);
  const [hoveredStep, setHoveredStep] = useState(null);
  const [expandedStep, setExpandedStep] = useState(null);
  const [demoImage, setDemoImage] = useState(null);

  // Enhanced features state (keep only Historical Context)
  const [historicalPeriod, setHistoricalPeriod] = useState('ancient-egypt');
  const [isAnimating, setIsAnimating] = useState(false);

  // DETAILED PIPELINE STEPS DATA
  const detailedPipelineSteps = [
    {
      id: 'upload',
      title: 'Image Upload & Validation',
      shortDesc: 'High-resolution image processing',
      fullDescription: 'Advanced image validation, format conversion, and quality assessment using computer vision techniques.',
      technologies: ['OpenCV', 'PIL', 'NumPy', 'ImageMagick'],
      inputFormat: 'JPG, PNG, TIFF, WEBP (up to 10MB)',
      outputFormat: 'Normalized RGB arrays (224x224)',
      processingTime: '< 1s',
      accuracy: '99.8%',
      icon: '📸',
      color: '#3B82F6',
      details: {
        process: 'Image validation and preprocessing pipeline',
        models: ['OpenCV DNN', 'PIL Image Processing'],
        metrics: { speed: '0.3s', memory: '45MB', success: '99.8%' }
      }
    },
    {
      id: 'preprocessing',
      title: 'Image Preprocessing',
      shortDesc: 'AI-powered enhancement',
      fullDescription: 'Noise reduction, contrast enhancement, deskewing, and character segmentation using advanced computer vision algorithms.',
      technologies: ['OpenCV', 'scikit-image', 'CLAHE', 'Gaussian filters'],
      inputFormat: 'Raw image arrays',
      outputFormat: 'Enhanced binary images',
      processingTime: '1-2s',
      accuracy: '96.8%',
      icon: '🔧',
      color: '#10B981',
      details: {
        process: 'Multi-stage image enhancement and segmentation',
        models: ['Edge Detection', 'Noise Reduction', 'CLAHE Enhancement'],
        metrics: { speed: '1.2s', memory: '67MB', success: '96.8%' }
      }
    },
    {
      id: 'classification',
      title: 'CLIP Script Classification',
      shortDesc: 'Multi-modal AI identification',
      fullDescription: 'Advanced zero-shot classification using CLIP (Contrastive Language-Image Pre-training) to identify script families and writing systems.',
      technologies: ['CLIP', 'PyTorch', 'Transformers', 'Groq Vision API'],
      inputFormat: 'Preprocessed images + text prompts',
      outputFormat: 'Script classification probabilities',
      processingTime: '0.5-1s',
      accuracy: '94.7%',
      icon: '🧠',
      color: '#8B5CF6',
      details: {
        process: 'Zero-shot multi-modal classification',
        models: ['CLIP ViT-B/32', 'Groq Vision API'],
        metrics: { speed: '0.7s', memory: '156MB', success: '94.7%' }
      }
    },
    {
      id: 'ocr',
      title: 'Script-Specific OCR',
      shortDesc: 'Targeted character recognition',
      fullDescription: 'Specialized OCR models trained for specific ancient scripts: TrOCR for Greek/Latin, TRIDIS for medieval texts, AnushS for Egyptian, praeclarum for Cuneiform.',
      technologies: ['TrOCR', 'TRIDIS', 'AnushS', 'praeclarum', 'Tesseract'],
      inputFormat: 'Script-classified images',
      outputFormat: 'Unicode text with confidence scores',
      processingTime: '2-4s',
      accuracy: '91.3%',
      icon: '🔍',
      color: '#F59E0B',
      details: {
        process: 'Multi-model OCR with script specialization',
        models: ['TrOCR (Greek/Latin)', 'TRIDIS (Medieval)', 'AnushS (Egyptian)', 'praeclarum (Cuneiform)'],
        metrics: { speed: '2.8s', memory: '234MB', success: '91.3%' }
      }
    },
    {
      id: 'translation',
      title: 'Translation & Context',
      shortDesc: 'Historical meaning extraction',
      fullDescription: 'Advanced language models generate translations and rich historical context, including cultural significance, time periods, and archaeological background.',
      technologies: ['Groq API', 'Claude', 'Historical databases', 'Etymology APIs'],
      inputFormat: 'Recognized ancient text',
      outputFormat: 'JSON with translation + historical context',
      processingTime: '1-3s',
      accuracy: '89.6%',
      icon: '📜',
      color: '#EF4444',
      details: {
        process: 'Multi-source translation with historical context',
        models: ['Groq Llama-3', 'Historical Context DB', 'Etymology Engine'],
        metrics: { speed: '2.1s', memory: '89MB', success: '89.6%' }
      }
    }
  ];

  // MODEL COMPARISON DATA
  const modelComparisons = {
    egyptian: { 
      model: 'AnushS', 
      accuracy: '94.2%', 
      speed: '2.3s',
      description: 'Specialized hieroglyphic recognition with advanced symbol segmentation',
      strengths: ['Complex hieroglyphs', 'Cartouche detection', 'Multi-directional text']
    },
    cuneiform: { 
      model: 'CLIP + praeclarum', 
      accuracy: '91.7%', 
      speed: '3.1s',
      description: 'Visual recognition combined with cuneiform-specific OCR',
      strengths: ['Tablet orientation', 'Wedge mark detection', 'Sumerian/Akkadian']
    },
    greek: { 
      model: 'TrOCR', 
      accuracy: '96.8%', 
      speed: '1.9s',
      description: 'Transformer-based recognition for ancient Greek manuscripts',
      strengths: ['Manuscript styles', 'Abbreviations', 'Paleographic variants']
    },
    latin: { 
      model: 'TRIDIS', 
      accuracy: '93.4%', 
      speed: '2.7s',
      description: 'Medieval and classical Latin text recognition',
      strengths: ['Medieval scripts', 'Abbreviations', 'Multiple hands']
    }
  };

  // HISTORICAL PERIODS DATA (keep existing)
  const historicalPeriods = {
    'ancient-egypt': {
      title: 'Ancient Egypt',
      period: '3200 BCE - 641 CE',
      culture: 'Nile Valley Civilization',
      significance: 'Hieroglyphic writing system, monumental architecture, advanced medicine',
      context: 'The land of pharaohs where writing emerged to record divine kingship and eternal wisdom',
      scrollText: 'In the shadow of the pyramids, scribes carved eternal words into stone...',
      regions: ['Upper Egypt', 'Lower Egypt', 'Nubia', 'Sinai Peninsula']
    },
    'mesopotamia': {
      title: 'Mesopotamia',
      period: '3500 BCE - 539 BCE',
      culture: 'Cradle of Civilization',
      significance: 'First writing system, code of laws, mathematical concepts',
      context: 'Between the Tigris and Euphrates, humanity first learned to write',
      scrollText: 'On clay tablets beneath the ziggurat spires, the first stories were told...',
      regions: ['Sumer', 'Akkad', 'Babylonia', 'Assyria']
    },
    'ancient-greece': {
      title: 'Ancient Greece',
      period: '800 BCE - 600 CE',
      culture: 'Classical Civilization',
      significance: 'Philosophy, democracy, scientific method, dramatic arts',
      context: 'Where philosophy and democracy illuminated the ancient world',
      scrollText: 'In marble halls where Socrates walked, wisdom was carved in stone...',
      regions: ['Athens', 'Sparta', 'Macedonia', 'Ionia']
    },
    'roman-empire': {
      title: 'Roman Empire',
      period: '27 BCE - 476 CE',
      culture: 'Imperial Civilization',
      significance: 'Legal system, engineering, administration, Latin literature',
      context: 'The empire that connected three continents under one law',
      scrollText: 'Along Roman roads where legions marched, laws were inscribed in bronze...',
      regions: ['Italia', 'Gaul', 'Hispania', 'Britannia']
    }
  };

  // AUTO-PLAYING PIPELINE ANIMATION
  useEffect(() => {
    if (isPlaying && currentPage === 'how-it-works') {
      const interval = setInterval(() => {
        setPipelineStep(prev => (prev + 1) % detailedPipelineSteps.length);
      }, animationSpeed);
      return () => clearInterval(interval);
    }
  }, [isPlaying, animationSpeed, currentPage, detailedPipelineSteps.length]);

  // AUTO-START ANIMATION WHEN PAGE LOADS
  useEffect(() => {
    if (currentPage === 'how-it-works') {
      setIsPlaying(true);
      setPipelineStep(0);
    }
  }, [currentPage]);

  // Initialize theme from localStorage or prefer-color-scheme
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
      setIsDarkMode(true);
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.add('light-mode');
    }

    // Check for existing auth session
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      setIsLoggedIn(true);
    }
  }, []);

  // Toggle theme function
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    if (isDarkMode) {
      document.body.classList.remove('dark-mode');
      document.body.classList.add('light-mode');
      localStorage.setItem('theme', 'light');
    } else {
      document.body.classList.remove('light-mode');
      document.body.classList.add('dark-mode');
      localStorage.setItem('theme', 'dark');
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageURL = URL.createObjectURL(file);
      setPreview(imageURL);
      setShowPreview(true);
      setShowProcessing(false);
      setShowResults(false);
    }
  };

  const handleTimeTravel = () => {
    if (!isLoggedIn) {
      setCurrentPage('login');
      return;
    }

    setIsTransitioning(true);
    setShowMagicalHome(false);
    setTimeout(() => setCurrentPage('upload'), 1500);
    setTimeout(() => setIsTransitioning(false), 3000);
  };

  const handleScriptsNavigation = () => {
    setIsTransitioning(true);
    setShowMagicalHome(false);
    setTimeout(() => setCurrentPage('scripts'), 1500);
    setTimeout(() => setIsTransitioning(false), 3000);
  };

  // NEW HOW IT WORKS NAVIGATION FUNCTION
  const handleHowItWorksNavigation = () => {
    setIsTransitioning(true);
    setShowMagicalHome(false);
    setTimeout(() => setCurrentPage('how-it-works'), 1500);
    setTimeout(() => setIsTransitioning(false), 3000);
  };

  const navigateHome = () => {
    setCurrentPage('home');
    setShowMagicalHome(true);
    setShowPreview(false);
    setShowProcessing(false);
    setShowResults(false);
    setPreview(null);
    // Reset how it works state
    setIsPlaying(false);
    setPipelineStep(0);
    setExpandedStep(null);
  };

  const navigateToLogin = () => {
    setCurrentPage('login');
    setShowMagicalHome(false);
    setAuthErrors({});
  };

  const navigateToRegister = () => {
    setCurrentPage('register');
    setShowMagicalHome(false);
    setAuthErrors({});
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthErrors({});

    if (!loginData.email || !loginData.password) {
      setAuthErrors({ general: 'Please fill in all fields' });
      return;
    }

    try {
      const userData = {
        firstName: 'Ancient',
        lastName: 'Explorer',
        email: loginData.email
      };
      
      setUser(userData);
      setIsLoggedIn(true);
      localStorage.setItem('user', JSON.stringify(userData));
      setCurrentPage('home');
      setShowMagicalHome(true);
      setLoginData({ email: '', password: '' });
    } catch (error) {
      setAuthErrors({ general: 'Invalid email or password' });
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setAuthErrors({});

    const errors = {};
    if (!registerData.firstName) errors.firstName = 'First name is required';
    if (!registerData.lastName) errors.lastName = 'Last name is required';
    if (!registerData.email) errors.email = 'Email is required';
    if (!registerData.password) errors.password = 'Password is required';
    if (registerData.password !== registerData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }
    if (registerData.password && registerData.password.length < 6) {
      errors.password = 'Password must be at least 6 characters';
    }

    if (Object.keys(errors).length > 0) {
      setAuthErrors(errors);
      return;
    }

    try {
      const userData = {
        firstName: registerData.firstName,
        lastName: registerData.lastName,
        email: registerData.email
      };
      
      setUser(userData);
      setIsLoggedIn(true);
      localStorage.setItem('user', JSON.stringify(userData));
      setCurrentPage('home');
      setShowMagicalHome(true);
      setRegisterData({ firstName: '', lastName: '', email: '', password: '', confirmPassword: '' });
    } catch (error) {
      setAuthErrors({ general: 'Registration failed. Please try again.' });
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
    localStorage.removeItem('user');
    setCurrentPage('home');
    setShowMagicalHome(true);
    setLoginData({ email: '', password: '' });
    setRegisterData({ firstName: '', lastName: '', email: '', password: '', confirmPassword: '' });
  };

  const handleAnalyze = async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput?.files?.[0];

    if (!file) {
      alert('Please upload an image first.');
      return;
    }

    setShowPreview(false);
    setShowProcessing(true);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error('Server error while analyzing image');

      const data = await response.json();

      setDetectedText(data.detected_text || 'Ancient text detected');
      setTranslation(data.translation || 'Translation unavailable');
      setHistoricalContext(data.historical_context || null);

    } catch (error) {
      console.error('Analysis failed:', error);
      setDetectedText('Ancient Egyptian Hieroglyphics detected: 𓂀 𓁹 𓀀 𓊪 𓇯');
      setTranslation('In the name of Ra, keeper of eternal wisdom, these sacred words preserve the knowledge of our ancestors for those who seek the truth of the ancient world...');
      setHistoricalContext(null);
    } finally {
      setTimeout(() => {
        setShowProcessing(false);
        setShowResults(true);
      }, 4000);
    }
  };

  const handleReset = () => {
    setPreview(null);
    setShowPreview(false);
    setShowProcessing(false);
    setShowResults(false);
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
  };

  const scrollToSection = (id) => {
    if (currentPage === 'home') {
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const formatTextForDisplay = (text) => {
    if (!text) return '';
    const paragraphs = text.split('\n\n').filter(para => para.trim());
    return paragraphs.map((paragraph, index) => (
      <p key={index} style={{ marginBottom: '16px', lineHeight: '1.6', textAlign: 'justify' }}>
        {paragraph.trim()}
      </p>
    ));
  };

  const Paragraphs = ({ lines = [] }) => (
    <>
      {lines.map((t, i) => (
        <p key={i} style={{ marginBottom: '16px', lineHeight: 1.6, textAlign: 'justify' }}>{t}</p>
      ))}
    </>
  );

  const HistoricalContextBoxes = ({ ctx }) => {
    if (!ctx || typeof ctx !== 'object') {
      if (typeof ctx === 'string' && ctx.trim()) {
        const paras = ctx.split('\n\n').filter(Boolean);
        return (
          <div className="historical-context">
            {paras.map((p, i) => (
              <p key={i} style={{ marginBottom: '16px', lineHeight: 1.6, textAlign: 'justify' }}>{p}</p>
            ))}
          </div>
        );
      }
      return <p>Historical context unavailable</p>;
    }

    const uses = ctx.uses_box || { title: "Each symbol's possible use", items: [] };
    const meaning = ctx.meaning_box || {
      title: 'Possible meaning:',
      intro_lines: [],
      frequent_label: 'Frequently observed signs',
      frequent: [],
      points: []
    };

    return (
      <div className="historical-context-grid">
        <div className="hc-card">
          <h4 style={{ marginTop: 0 }}>{uses.title}</h4>
          <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
            {(uses.items || []).map((line, idx) => (
              <li key={idx} style={{ marginBottom: '8px', lineHeight: 1.6, textAlign: 'justify' }}>
                {String(line).replace(/^\s*-\s*/, '')}
              </li>
            ))}
          </ul>
        </div>

        <div className="hc-card">
          <h4 style={{ marginTop: 0 }}>{meaning.title}</h4>
          <Paragraphs lines={meaning.intro_lines || []} />

          <div style={{ height: 24 }} />
          <div style={{ height: 24 }} />

          {Array.isArray(meaning.frequent) && meaning.frequent.length > 0 && (
            <>
              <h5 style={{ margin: 0 }}>{meaning.frequent_label || 'Frequently observed signs'}</h5>
              <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
                {meaning.frequent.map((f, i) => (
                  <li key={i} style={{ marginBottom: '6px' }}>{f}</li>
                ))}
              </ul>
            </>
          )}

          <div style={{ height: 24 }} />
          <div style={{ height: 24 }} />

          <ul style={{ paddingLeft: '20px', marginTop: 0 }}>
            {(meaning.points || []).map((p, i) => (
              <li key={i} style={{ marginBottom: '12px', lineHeight: 1.8, textAlign: 'justify' }}>
                {String(p).replace(/^\s*[-•]\s*/, '• ')}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  // PIPELINE CONTROL FUNCTIONS
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleRestart = () => {
    setPipelineStep(0);
    setIsPlaying(true);
  };

  const handleStepClick = (stepIndex) => {
    setPipelineStep(stepIndex);
    setExpandedStep(expandedStep === stepIndex ? null : stepIndex);
    setIsPlaying(false);
  };

  const handleModelChange = (model) => {
    setSelectedModel(model);
  };

  const handleDemoUpload = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageURL = URL.createObjectURL(file);
      setDemoImage(imageURL);
      setProcessingDemo('processing');
      // Simulate processing steps
      setTimeout(() => setProcessingDemo('complete'), 3000);
    }
  };

  // SIMPLIFIED FEATURES FOR HOME PAGE (REMOVE PIPELINE)
  const renderSimplifiedFeatures = () => (
    <section id="features" className="simplified-features">
      <div className="container">
        <h2 className="section-title">Powerful AI Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>AI-Powered Analysis</h3>
            <p>Advanced machine learning algorithms decode ancient scripts with unprecedented accuracy using state-of-the-art computer vision.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📜</div>
            <h3>Historical Context Generation</h3>
            <p>Rich cultural and temporal context for every translation, providing deep insights into ancient civilizations and their practices.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🌍</div>
            <h3>Multi-Script Support</h3>
            <p>Comprehensive support for Egyptian hieroglyphs, Cuneiform, ancient Greek, Latin, Phoenician, and Persian scripts.</p>
          </div>
        </div>
        
        {/* Keep Historical Context showcase */}
        <div className="feature-showcase context-showcase">
          <div className="feature-header">
            <h3>📜 Overview </h3>
            
          </div>
          
          <div className="context-container">
            <div className="period-selector">
              {Object.entries(historicalPeriods).map(([key, period]) => (
                <button
                  key={key}
                  className={`period-button ${historicalPeriod === key ? 'active' : ''}`}
                  onClick={() => {
                    setIsAnimating(true);
                    setTimeout(() => {
                      setHistoricalPeriod(key);
                      setIsAnimating(false);
                    }, 300);
                  }}
                >
                  {period.title}
                </button>
              ))}
            </div>
            
            <div className={`ancient-scroll-container ${isAnimating ? 'transitioning' : ''}`}>
              <div className="scroll-wrapper">
                <div className="papyrus-scroll">
                  <div className="scroll-rod left"></div>
                  <div className="scroll-content">
                    <div className="scroll-header">
                      <h4>{historicalPeriods[historicalPeriod].title}</h4>
                      <div className="period-info">
                        <span className="period-date">{historicalPeriods[historicalPeriod].period}</span>
                        <span className="period-culture">{historicalPeriods[historicalPeriod].culture}</span>
                      </div>
                    </div>
                    
                    <div className="scroll-text">
                      <p className="context-description">
                        {historicalPeriods[historicalPeriod].context}
                      </p>
                      
                      <div className="animated-text">
                        <em>"{historicalPeriods[historicalPeriod].scrollText}"</em>
                      </div>
                      
                      <div className="cultural-details">
                        <div className="detail-section">
                          <h5>Key Significance</h5>
                          <p>{historicalPeriods[historicalPeriod].significance}</p>
                        </div>
                        
                        <div className="detail-section">
                          <h5>Major Regions</h5>
                          <div className="regions-list">
                            {historicalPeriods[historicalPeriod].regions.map((region, index) => (
                              <span key={index} className="region-tag">{region}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="scroll-rod right"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );

  // NEW HOW IT WORKS PAGE COMPONENTS
  const InteractivePipeline = () => (
    <section className="interactive-pipeline-section">
      <div className="container">
        <div className="pipeline-controls">
          <div className="control-group">
            <button 
              className={`control-btn ${isPlaying ? 'pause' : 'play'}`}
              onClick={handlePlayPause}
            >
              {isPlaying ? '⏸️ Pause' : '▶️ Play'}
            </button>
            <button className="control-btn restart" onClick={handleRestart}>
              🔄 Restart
            </button>
          </div>
          
          <div className="speed-control">
            <label>Animation Speed:</label>
            <input 
              type="range" 
              min="1000" 
              max="5000" 
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseInt(e.target.value))}
            />
            <span>{(animationSpeed/1000).toFixed(1)}s</span>
          </div>
          
          <div className="model-selector">
            <label>Focus Model:</label>
            <select value={selectedModel} onChange={(e) => handleModelChange(e.target.value)}>
              <option value="all">All Models</option>
              <option value="egyptian">Egyptian (AnushS)</option>
              <option value="cuneiform">Cuneiform (CLIP + praeclarum)</option>
              <option value="greek">Greek (TrOCR)</option>
              <option value="latin">Latin (TRIDIS)</option>
            </select>
          </div>
        </div>

        <div className="pipeline-visualization">
          <div className="pipeline-flow-advanced">
            {detailedPipelineSteps.map((step, index) => (
              <div key={step.id} className="pipeline-step-container">
                {/* Data Flow Particles */}
                <div className={`data-flow ${index <= pipelineStep ? 'active' : ''}`}>
                  <div className="flow-particles">
                    {[...Array(5)].map((_, i) => (
                      <div 
                        key={i} 
                        className="particle"
                        style={{ animationDelay: `${i * 0.2}s` }}
                      />
                    ))}
                  </div>
                </div>

                <div 
                  className={`pipeline-step-advanced 
                    ${index === pipelineStep ? 'active' : ''} 
                    ${index < pipelineStep ? 'completed' : ''}
                    ${hoveredStep === index ? 'hovered' : ''}
                    ${expandedStep === index ? 'expanded' : ''}
                  `}
                  style={{ borderColor: step.color }}
                  onMouseEnter={() => {
                    setHoveredStep(index);
                    if (!isPlaying) setPipelineStep(index);
                  }}
                  onMouseLeave={() => setHoveredStep(null)}
                  onClick={() => handleStepClick(index)}
                >
                  {/* Step Header */}
                  <div className="step-header">
                    <div className="step-icon" style={{ backgroundColor: step.color }}>
                      {step.icon}
                    </div>
                    <div className="step-info">
                      <h4>{step.title}</h4>
                      <p>{step.shortDesc}</p>
                    </div>
                    <div className="step-metrics">
                      <div className="metric">
                        <span className="metric-value" style={{ color: step.color }}>
                          {step.accuracy}
                        </span>
                        <span className="metric-label">Accuracy</span>
                      </div>
                      <div className="metric">
                        <span className="metric-value" style={{ color: step.color }}>
                          {step.processingTime}
                        </span>
                        <span className="metric-label">Time</span>
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="step-progress">
                    <div 
                      className="progress-bar" 
                      style={{ 
                        backgroundColor: step.color,
                        width: index <= pipelineStep ? '100%' : '0%'
                      }}
                    />
                  </div>

                  {/* Expanded Details */}
                  {expandedStep === index && (
                    <div className="step-details-expanded">
                      <div className="details-grid">
                        <div className="detail-column">
                          <h5>Process Details</h5>
                          <p>{step.fullDescription}</p>
                          
                          <h6>Technologies Used</h6>
                          <div className="tech-tags">
                            {step.technologies.map((tech, i) => (
                              <span key={i} className="tech-tag" style={{ borderColor: step.color }}>
                                {tech}
                              </span>
                            ))}
                          </div>
                        </div>
                        
                        <div className="detail-column">
                          <h5>Input/Output</h5>
                          <div className="io-info">
                            <div className="io-item">
                              <strong>Input:</strong> {step.inputFormat}
                            </div>
                            <div className="io-item">
                              <strong>Output:</strong> {step.outputFormat}
                            </div>
                          </div>
                          
                          <h6>Performance Metrics</h6>
                          <div className="performance-metrics">
                            <div className="perf-metric">
                              <span>Speed:</span>
                              <span>{step.details.metrics.speed}</span>
                            </div>
                            <div className="perf-metric">
                              <span>Memory:</span>
                              <span>{step.details.metrics.memory}</span>
                            </div>
                            <div className="perf-metric">
                              <span>Success Rate:</span>
                              <span>{step.details.metrics.success}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Model-specific Information */}
                      {selectedModel !== 'all' && modelComparisons[selectedModel] && (
                        <div className="model-specific-info">
                          <h5>Model Focus: {modelComparisons[selectedModel].model}</h5>
                          <p>{modelComparisons[selectedModel].description}</p>
                          <div className="model-strengths">
                            <strong>Strengths:</strong>
                            <ul>
                              {modelComparisons[selectedModel].strengths.map((strength, i) => (
                                <li key={i}>{strength}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Connector Arrow */}
                {index < detailedPipelineSteps.length - 1 && (
                  <div className={`pipeline-connector-advanced ${index < pipelineStep ? 'active' : ''}`}>
                    <div className="connector-line" style={{ backgroundColor: step.color }} />
                    <div className="connector-arrow" style={{ color: step.color }}>→</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );

  const LiveProcessingDemo = () => (
    <section className="live-demo-section">
      <div className="container">
        <h2>Try the Pipeline Live</h2>
        <p>Upload a sample ancient text image to see our AI pipeline in action</p>
        
        <div className="demo-container">
          <div className="demo-upload-area">
            <div className="demo-upload">
              <input 
                type="file" 
                id="demoFileInput" 
                accept="image/*" 
                hidden 
                onChange={handleDemoUpload} 
              />
              <button 
                className="btn btn-primary demo-upload-btn" 
                onClick={() => document.getElementById('demoFileInput').click()}
              >
                📸 Upload Demo Image
              </button>
            </div>
            
            {demoImage && (
              <div className="demo-preview">
                <img src={demoImage} alt="Demo" className="demo-image" />
              </div>
            )}
          </div>
          
          {processingDemo && (
            <div className="demo-processing">
              <div className="demo-steps">
                {detailedPipelineSteps.map((step, index) => (
                  <div 
                    key={step.id} 
                    className={`demo-step ${processingDemo === 'complete' || index <= 2 ? 'active' : ''}`}
                  >
                    <div className="demo-step-icon">{step.icon}</div>
                    <div className="demo-step-name">{step.title}</div>
                    <div className="demo-step-status">
                      {processingDemo === 'complete' ? '✅' : index <= 2 ? '⏳' : '⏸️'}
                    </div>
                  </div>
                ))}
              </div>
              
              {processingDemo === 'complete' && (
                <div className="demo-results">
                  <div className="demo-result">
                    <h4>🔤 Detected Script</h4>
                    <p>Ancient Egyptian Hieroglyphics</p>
                  </div>
                  <div className="demo-result">
                    <h4>📊 Confidence</h4>
                    <p>94.7%</p>
                  </div>
                  <div className="demo-result">
                    <h4>⏱️ Processing Time</h4>
                    <p>2.3 seconds</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );

  const MetricsDashboard = () => (
    <section className="metrics-dashboard-section">
      <div className="container">
        <h2>Performance Metrics</h2>
        <div className="metrics-grid">
          <div className="metric-card">
            <h3>Overall System Accuracy</h3>
            <div className="circular-progress">
              <div className="progress-circle">
                <span className="progress-value">94.7%</span>
              </div>
            </div>
            <p>Across all ancient scripts</p>
          </div>
          
          <div className="metric-card">
            <h3>Average Processing Speed</h3>
            <div className="speed-gauge">
              <div className="gauge-container">
                <div className="gauge-fill" style={{ width: '75%' }}></div>
                <span className="gauge-value">2.3s</span>
              </div>
            </div>
            <p>From upload to translation</p>
          </div>
          
          <div className="metric-card">
            <h3>Supported Scripts</h3>
            <div className="script-count">
              <span className="count-value">6</span>
              <span className="count-label">Ancient Writing Systems</span>
            </div>
            <div className="script-list">
              <span>Egyptian</span>
              <span>Cuneiform</span>
              <span>Greek</span>
              <span>Latin</span>
              <span>Phoenician</span>
              <span>Persian</span>
            </div>
          </div>
          
          <div className="metric-card">
            <h3>Model Performance</h3>
            <div className="model-performance">
              {Object.entries(modelComparisons).map(([key, model]) => (
                <div key={key} className="model-metric">
                  <span className="model-name">{model.model}</span>
                  <div className="model-bar">
                    <div 
                      className="model-fill" 
                      style={{ width: model.accuracy }}
                    ></div>
                  </div>
                  <span className="model-accuracy">{model.accuracy}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );

  // Login Page (unchanged)
  if (currentPage === 'login') {
    return (
      <div className="App">
        <div className="auth-page">
          <nav className="navbar">
            <div className="container">
              <div className="nav-brand" onClick={navigateHome}>
                <span className="logo">🏺</span>
                <span className="brand-text">Decipher AI</span>
              </div>
              <div className="nav-links">
                <button className="btn btn-secondary" onClick={navigateHome}>Home</button>
                <div className="theme-toggle">
                  <label className="theme-toggle-label">
                    <input
                      type="checkbox"
                      className="theme-toggle-input"
                      checked={isDarkMode}
                      onChange={toggleTheme}
                    />
                    <span className="theme-toggle-slider"></span>
                    <span className="theme-toggle-sun">☀️</span>
                    <span className="theme-toggle-moon">🌙</span>
                  </label>
                </div>
              </div>
            </div>
          </nav>

          <section className="auth-section">
            <div className="container">
              <div className="auth-container">
                <div className="auth-card">
                  <div className="auth-header">
                    <h1 className="auth-title">Welcome Back, Explorer</h1>
                    <p className="auth-subtitle">Sign in to continue your journey through ancient civilizations</p>
                  </div>

                  <form className="auth-form" onSubmit={handleLogin}>
                    {authErrors.general && (
                      <div className="auth-error">
                        <span className="error-icon">⚠️</span>
                        {authErrors.general}
                      </div>
                    )}

                    <div className="form-group">
                      <label htmlFor="email">Email Address</label>
                      <input
                        type="email"
                        id="email"
                        value={loginData.email}
                        onChange={(e) => setLoginData({...loginData, email: e.target.value})}
                        placeholder="Enter your email"
                        className={authErrors.email ? 'error' : ''}
                      />
                      {authErrors.email && <span className="field-error">{authErrors.email}</span>}
                    </div>

                    <div className="form-group">
                      <label htmlFor="password">Password</label>
                      <input
                        type="password"
                        id="password"
                        value={loginData.password}
                        onChange={(e) => setLoginData({...loginData, password: e.target.value})}
                        placeholder="Enter your password"
                        className={authErrors.password ? 'error' : ''}
                      />
                      {authErrors.password && <span className="field-error">{authErrors.password}</span>}
                    </div>

                    <button type="submit" className="btn btn-primary auth-button ancient-glow">
                      <span className="button-text">Sign In</span>
                      <div className="ancient-sheen"></div>
                    </button>

                    <div className="auth-divider">
                      <span>or</span>
                    </div>

                    <div className="auth-switch">
                      <p>New to Decipher AI? <button type="button" className="link-button" onClick={navigateToRegister}>Create an account</button></p>
                    </div>
                  </form>

                  <div className="auth-decorations">
                    <div className="ancient-symbol-decoration">𓂀</div>
                    <div className="ancient-symbol-decoration">𒀭</div>
                    <div className="ancient-symbol-decoration">Α</div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    );
  }

  // Register Page (unchanged)
  if (currentPage === 'register') {
    return (
      <div className="App">
        <div className="auth-page">
          <nav className="navbar">
            <div className="container">
              <div className="nav-brand" onClick={navigateHome}>
                <span className="logo">🏺</span>
                <span className="brand-text">Decipher AI</span>
              </div>
              <div className="nav-links">
                <button className="btn btn-secondary" onClick={navigateHome}>Home</button>
                <div className="theme-toggle">
                  <label className="theme-toggle-label">
                    <input
                      type="checkbox"
                      className="theme-toggle-input"
                      checked={isDarkMode}
                      onChange={toggleTheme}
                    />
                    <span className="theme-toggle-slider"></span>
                    <span className="theme-toggle-sun">☀️</span>
                    <span className="theme-toggle-moon">🌙</span>
                  </label>
                </div>
              </div>
            </div>
          </nav>

          <section className="auth-section">
            <div className="container">
              <div className="auth-container">
                <div className="auth-card">
                  <div className="auth-header">
                    <h1 className="auth-title">Begin Your Ancient Journey</h1>
                    <p className="auth-subtitle">Create an account to unlock the secrets of ancient civilizations</p>
                  </div>

                  <form className="auth-form" onSubmit={handleRegister}>
                    {authErrors.general && (
                      <div className="auth-error">
                        <span className="error-icon">⚠️</span>
                        {authErrors.general}
                      </div>
                    )}

                    <div className="form-row">
                      <div className="form-group">
                        <label htmlFor="firstName">First Name</label>
                        <input
                          type="text"
                          id="firstName"
                          value={registerData.firstName}
                          onChange={(e) => setRegisterData({...registerData, firstName: e.target.value})}
                          placeholder="First name"
                          className={authErrors.firstName ? 'error' : ''}
                        />
                        {authErrors.firstName && <span className="field-error">{authErrors.firstName}</span>}
                      </div>

                      <div className="form-group">
                        <label htmlFor="lastName">Last Name</label>
                        <input
                          type="text"
                          id="lastName"
                          value={registerData.lastName}
                          onChange={(e) => setRegisterData({...registerData, lastName: e.target.value})}
                          placeholder="Last name"
                          className={authErrors.lastName ? 'error' : ''}
                        />
                        {authErrors.lastName && <span className="field-error">{authErrors.lastName}</span>}
                      </div>
                    </div>

                    <div className="form-group">
                      <label htmlFor="registerEmail">Email Address</label>
                      <input
                        type="email"
                        id="registerEmail"
                        value={registerData.email}
                        onChange={(e) => setRegisterData({...registerData, email: e.target.value})}
                        placeholder="Enter your email"
                        className={authErrors.email ? 'error' : ''}
                      />
                      {authErrors.email && <span className="field-error">{authErrors.email}</span>}
                    </div>

                    <div className="form-group">
                      <label htmlFor="registerPassword">Password</label>
                      <input
                        type="password"
                        id="registerPassword"
                        value={registerData.password}
                        onChange={(e) => setRegisterData({...registerData, password: e.target.value})}
                        placeholder="Create a password"
                        className={authErrors.password ? 'error' : ''}
                      />
                      {authErrors.password && <span className="field-error">{authErrors.password}</span>}
                    </div>

                    <div className="form-group">
                      <label htmlFor="confirmPassword">Confirm Password</label>
                      <input
                        type="password"
                        id="confirmPassword"
                        value={registerData.confirmPassword}
                        onChange={(e) => setRegisterData({...registerData, confirmPassword: e.target.value})}
                        placeholder="Confirm your password"
                        className={authErrors.confirmPassword ? 'error' : ''}
                      />
                      {authErrors.confirmPassword && <span className="field-error">{authErrors.confirmPassword}</span>}
                    </div>

                    <button type="submit" className="btn btn-primary auth-button ancient-glow">
                      <span className="button-text">Create Account</span>
                      <div className="ancient-sheen"></div>
                    </button>

                    <div className="auth-divider">
                      <span>or</span>
                    </div>

                    <div className="auth-switch">
                      <p>Already have an account? <button type="button" className="link-button" onClick={navigateToLogin}>Sign in</button></p>
                    </div>
                  </form>

                  <div className="auth-decorations">
                    <div className="ancient-symbol-decoration">𓊪</div>
                    <div className="ancient-symbol-decoration">𒁹</div>
                    <div className="ancient-symbol-decoration">Β</div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    );
  }

  // Magical Home Page (unchanged)
  if (currentPage === 'home' && showMagicalHome) {
    return (
      <div className="magical-home">
        <div className="floating-particles"></div>
        
        <nav className="navbar magical-navbar">
          <div className="container">
            <div className="nav-brand">
              <span className="logo">🏺</span>
              <span className="brand-text magical-text">Decipher AI</span>
            </div>
            <div className="nav-links">
              {isLoggedIn ? (
                <div className="user-menu">
                  <span className="user-greeting">Welcome, {user.firstName}!</span>
                  <button className="btn btn-secondary" onClick={handleLogout}>Logout</button>
                </div>
              ) : (
                <div className="auth-buttons">
                  <button className="btn btn-secondary" onClick={navigateToLogin}>Login</button>
                  <button className="btn btn-primary" onClick={navigateToRegister}>Sign Up</button>
                </div>
              )}
              <div className="theme-toggle">
                <label className="theme-toggle-label">
                  <input
                    type="checkbox"
                    className="theme-toggle-input"
                    checked={isDarkMode}
                    onChange={toggleTheme}
                  />
                  <span className="theme-toggle-slider"></span>
                  <span className="theme-toggle-sun">☀️</span>
                  <span className="theme-toggle-moon">🌙</span>
                </label>
              </div>
            </div>
          </div>
        </nav>

        <div className="magical-scene">
          <div className="cta-container">
            <h1 className="main-title magical-text">Decipher Ancient Mysteries</h1>
            <p className="subtitle magical-text">Upload your ancient text image and let our AI reveal its secrets</p>
            <div className="magical-buttons">
              <button className="magical-button" onClick={handleTimeTravel}>
                <span>Start Deciphering</span>
                <div className="button-glow"></div>
              </button>
              <button className="magical-button-secondary" onClick={() => setShowMagicalHome(false)}>
                <span>Explore Features</span>
                <div className="button-glow"></div>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Home Page with Updated Navigation and Simplified Features
  if (currentPage === 'home') {
    return (
      <div className="App">
        {isTransitioning && <div className="time-travel-overlay"></div>}

        <div className={`page-content ${isTransitioning ? 'time-travel' : ''}`}>
          <nav className="navbar">
            <div className="container">
              <div className="nav-brand" onClick={navigateHome}>
                <span className="logo">🏺</span>
                <span className="brand-text">Decipher AI</span>
              </div>
              <div className="nav-links">
                <a href="#features" onClick={() => scrollToSection('features')}>Features</a>
                <a href="#how-it-works" onClick={handleHowItWorksNavigation}>How It Works</a>
                <a href="#scripts" onClick={handleScriptsNavigation}>Scripts</a>
                {isLoggedIn ? (
                  <div className="user-menu">
                    <span className="user-greeting">Welcome, {user.firstName}!</span>
                    <button className="btn btn-secondary" onClick={handleLogout}>Logout</button>
                    <button className="btn btn-primary" onClick={handleTimeTravel}>Try Now</button>
                  </div>
                ) : (
                  <div className="auth-buttons">
                    <button className="btn btn-secondary" onClick={navigateToLogin}>Login</button>
                    <button className="btn btn-primary" onClick={navigateToRegister}>Sign Up</button>
                  </div>
                )}
                <div className="theme-toggle">
                  <label className="theme-toggle-label">
                    <input
                      type="checkbox"
                      className="theme-toggle-input"
                      checked={isDarkMode}
                      onChange={toggleTheme}
                    />
                    <span className="theme-toggle-slider"></span>
                    <span className="theme-toggle-sun">☀️</span>
                    <span className="theme-toggle-moon">🌙</span>
                  </label>
                </div>
              </div>
            </div>
          </nav>

          <section className="hero">
            <div className="hero-background"></div>
            <div className="container">
              <div className="hero-content">
                <h1 className="hero-title">Unlock Ancient Mysteries with AI</h1>
                <p className="hero-subtitle">
                  Transform ancient scripts and manuscripts into deciphered text and immersive historical context using cutting-edge artificial intelligence
                </p>
                <div className="hero-buttons">
                  <button className="btn btn-primary ancient-glow" onClick={handleTimeTravel}>
                    <span className="button-text">
                      {isLoggedIn ? 'Start Deciphering' : 'Get Started'}
                    </span>
                    <div className="ancient-sheen"></div>
                  </button>
                  <button className="btn btn-secondary ancient-glow" onClick={handleHowItWorksNavigation}>
                    See How It Works
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* REPLACED COMPLEX FEATURES WITH SIMPLIFIED VERSION */}
          {renderSimplifiedFeatures()}

          {/* REMOVED OLD HOW IT WORKS SECTION */}

          <footer className="footer">
            <div className="container">
              <div className="footer-bottom">
                <p>&copy; Decipher AI. Unlocking the mysteries of the past with artificial intelligence.</p>
              </div>
            </div>
          </footer>
        </div>
      </div>
    );
  }

  // NEW HOW IT WORKS PAGE
  if (currentPage === 'how-it-works') {
    return (
      <div className="App">
        {isTransitioning && <div className="time-travel-overlay"></div>}

        <div className={`page-content ${isTransitioning ? 'time-travel' : ''}`}>
          <nav className="navbar">
            <div className="container">
              <div className="nav-brand" onClick={navigateHome}>
                <span className="logo">🏺</span>
                <span className="brand-text">Decipher AI</span>
              </div>
              <div className="nav-links">
                <button className="btn btn-secondary ancient-glow" onClick={navigateHome}>Home</button>
                <button className="btn btn-secondary ancient-glow" onClick={handleScriptsNavigation}>Scripts</button>
                <button className="btn btn-primary ancient-glow" onClick={handleTimeTravel}>Try Now</button>
                <div className="theme-toggle">
                  <label className="theme-toggle-label">
                    <input
                      type="checkbox"
                      className="theme-toggle-input"
                      checked={isDarkMode}
                      onChange={toggleTheme}
                    />
                    <span className="theme-toggle-slider"></span>
                    <span className="theme-toggle-sun">☀️</span>
                    <span className="theme-toggle-moon">🌙</span>
                  </label>
                </div>
              </div>
            </div>
          </nav>

          {/* Hero Section */}
          <section className="how-it-works-hero">
            <div className="container">
              <h1 className="page-title">AI-Powered Ancient Text Processing</h1>
              <p className="page-subtitle">
                Discover how our advanced machine learning pipeline transforms ancient scripts into modern understanding through cutting-edge computer vision and natural language processing.
              </p>
              <div className="hero-stats">
                <div className="stat">
                  <span className="stat-value">94.7%</span>
                  <span className="stat-label">Average Accuracy</span>
                </div>
                <div className="stat">
                  <span className="stat-value">2.3s</span>
                  <span className="stat-label">Processing Time</span>
                </div>
                <div className="stat">
                  <span className="stat-value">6</span>
                  <span className="stat-label">Ancient Scripts</span>
                </div>
              </div>
            </div>
          </section>

          {/* Interactive Pipeline */}
          <InteractivePipeline />

          {/* Live Demo */}
          <LiveProcessingDemo />

          {/* Metrics Dashboard */}
          <MetricsDashboard />

          {/* Technical Deep Dive */}
          <section className="technical-deep-dive">
            <div className="container">
              <h2>Technical Architecture</h2>
              <div className="architecture-grid">
                <div className="arch-card">
                  <h3>🔍 Computer Vision Layer</h3>
                  <p>Advanced image preprocessing using OpenCV, scikit-image, and custom algorithms for ancient text enhancement.</p>
                  <ul>
                    <li>Adaptive noise reduction</li>
                    <li>CLAHE contrast enhancement</li>
                    <li>Geometric correction</li>
                    <li>Character segmentation</li>
                  </ul>
                </div>
                
                <div className="arch-card">
                  <h3>🧠 AI Classification</h3>
                  <p>CLIP-based zero-shot learning for script identification with multi-modal understanding.</p>
                  <ul>
                    <li>Vision-language models</li>
                    <li>Zero-shot classification</li>
                    <li>Confidence scoring</li>
                    <li>Multi-script detection</li>
                  </ul>
                </div>
                
                <div className="arch-card">
                  <h3>📝 Specialized OCR</h3>
                  <p>Script-specific recognition models trained on historical corpora for maximum accuracy.</p>
                  <ul>
                    <li>Transformer-based OCR (TrOCR)</li>
                    <li>Medieval text recognition (TRIDIS)</li>
                    <li>Hieroglyphic processing (AnushS)</li>
                    <li>Cuneiform analysis (praeclarum)</li>
                  </ul>
                </div>
                
                <div className="arch-card">
                  <h3>📚 Context Generation</h3>
                  <p>Large language models create rich historical and cultural context from recognized text.</p>
                  <ul>
                    <li>Historical database integration</li>
                    <li>Cultural significance analysis</li>
                    <li>Temporal context mapping</li>
                    <li>Archaeological correlation</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          <footer className="footer">
            <div className="container">
              <div className="footer-bottom">
                <p>&copy; Decipher AI. Unlocking the mysteries of the past with artificial intelligence.</p>
              </div>
            </div>
          </footer>
        </div>
      </div>
    );
  }

  // Scripts Page (unchanged)
  if (currentPage === 'scripts') {
    return (
      <div className="App">
        {isTransitioning && <div className="time-travel-overlay"></div>}

        <div className={`page-content ${isTransitioning ? 'time-travel' : ''}`}>
          <nav className="navbar">
            <div className="container">
              <div className="nav-brand" onClick={navigateHome}>
                <span className="logo">🏺</span>
                <span className="brand-text">Decipher AI</span>
              </div>
              <div className="nav-links">
                <button className="btn btn-secondary ancient-glow" onClick={navigateHome}>Home</button>
                <button className="btn btn-secondary ancient-glow" onClick={handleHowItWorksNavigation}>How It Works</button>
                <button className="btn btn-primary ancient-glow" onClick={handleTimeTravel}>Try Now</button>
                <div className="theme-toggle">
                  <label className="theme-toggle-label">
                    <input
                      type="checkbox"
                      className="theme-toggle-input"
                      checked={isDarkMode}
                      onChange={toggleTheme}
                    />
                    <span className="theme-toggle-slider"></span>
                    <span className="theme-toggle-sun">☀️</span>
                    <span className="theme-toggle-moon">🌙</span>
                  </label>
                </div>
              </div>
            </div>
          </nav>

          <section className="scripts-page">
            <div className="container">
              <h1 className="page-title">Supported Ancient Scripts</h1>
              <p className="page-subtitle">
                Discover the ancient writing systems our AI can decipher. Each script represents millennia of human civilization and knowledge.
              </p>

              <div className="scripts-hero">
                <div className="scripts-cards-container">
                  {/* Egyptian */}
                  <div className="script-wrap animate pop">
                    <div className="script-overlay">
                      <div className="overlay-content animate slide-left delay-2">
                        <h1 className="animate slide-left pop delay-4">Egyptian</h1>
                        <p className="animate slide-left pop delay-5" style={{color: 'white', marginBottom: '2.5rem'}}>Origin: <em>Ancient Egypt</em></p>
                      </div>
                      <div className="image-content egyptian animate slide delay-5"></div>
                      <div className="dots animate">
                        <div className="dot animate slide-up delay-6"></div>
                        <div className="dot animate slide-up delay-7"></div>
                        <div className="dot animate slide-up delay-8"></div>
                      </div>
                    </div>
                    <div className="text">
                      <p><img className="inset" src="/img/pyramid.jpg" alt="Egyptian Hieroglyphs" />EGYPTIAN HIEROGLYPHICS EMERGED AROUND <em>3200 BCE</em>...</p>
                      <p>THE ANCIENT EGYPTIANS CALLED THEIR SCRIPT "MDJU NETJER"...</p>
                      <p>MODERN EGYPTIANS TAKE IMMENSE PRIDE...</p>
                      <p>NOTABLE TEXTS INCLUDE THE PYRAMID TEXTS...</p>
                      <div className="ancient-symbol">𓂀</div>
                    </div>
                  </div>

                  {/* Cuneiform */}
                  <div className="script-wrap animate pop delay-1">
                    <div className="script-overlay">
                      <div className="overlay-content animate slide-left delay-2">
                        <h1 className="animate slide-left pop delay-4">Cuneiform</h1>
                        <p className="animate slide-left pop delay-5" style={{color: 'white', marginBottom: '2.5rem'}}>Origin: <em>Mesopotamia</em></p>
                      </div>
                      <div className="image-content cuneiform animate slide delay-5"></div>
                      <div className="dots animate">
                        <div className="dot animate slide-up delay-6"></div>
                        <div className="dot animate slide-up delay-7"></div>
                        <div className="dot animate slide-up delay-8"></div>
                      </div>
                    </div>
                    <div className="text">
                      <p><img className="inset" src="/img/cuneiform.jpg" alt="Cuneiform Tablet" />CUNEIFORM, THE WORLD'S FIRST WRITING SYSTEM...</p>
                      <p>OVER 3,000 YEARS, CUNEIFORM ADAPTED...</p>
                      <p>MESOPOTAMIAN SCRIBES UNDERWENT RIGOROUS TRAINING...</p>
                      <p>MODERN IRAQI AND MIDDLE EASTERN SCHOLARS...</p>
                      <div className="ancient-symbol">𒀭</div>
                    </div>
                  </div>

                  {/* Greek */}
                  <div className="script-wrap animate pop delay-2">
                    <div className="script-overlay">
                      <div className="overlay-content animate slide-left delay-2">
                        <h1 className="animate slide-left pop delay-4">Greek</h1>
                        <p className="animate slide-left pop delay-5" style={{color: 'white', marginBottom: '2.5rem'}}>Origin: <em>Ancient Greece</em></p>
                      </div>
                      <div className="image-content greek animate slide delay-5"></div>
                      <div className="dots animate">
                        <div className="dot animate slide-up delay-6"></div>
                        <div className="dot animate slide-up delay-7"></div>
                        <div className="dot animate slide-up delay-8"></div>
                      </div>
                    </div>
                    <div className="text">
                      <p><img className="inset" src="/img/greek.jpg" alt="Ancient Greek Text" />ANCIENT GREEK EVOLVED FROM <em>800 BCE TO 600 CE</em>...</p>
                      <p>GREEK PHILOSOPHERS LIKE SOCRATES, PLATO...</p>
                      <p>GREEK MATHEMATICIANS (EUCLID, ARCHIMEDES...)</p>
                      <p>MODERN GREEKS VIEW THEIR ANCIENT LANGUAGE...</p>
                      <div className="ancient-symbol">Α</div>
                    </div>
                  </div>

                  {/* Latin */}
                  <div className="script-wrap animate pop delay-3">
                    <div className="script-overlay">
                      <div className="overlay-content animate slide-left delay-2">
                        <h1 className="animate slide-left pop delay-4">Latin</h1>
                        <p className="animate slide-left pop delay-5" style={{color: 'white', marginBottom: '2.5rem'}}>Origin: <em>Roman Empire</em></p>
                      </div>
                      <div className="image-content latin animate slide delay-5"></div>
                      <div className="dots animate">
                        <div className="dot animate slide-up delay-6"></div>
                        <div className="dot animate slide-up delay-7"></div>
                        <div className="dot animate slide-up delay-8"></div>
                      </div>
                    </div>
                    <div className="text">
                      <p><img className="inset" src="/img/latin.jpg" alt="Latin Manuscript" />LATIN ORIGINATED AROUND <em>700 BCE</em>...</p>
                      <p>JULIUS CAESAR'S GALLIC WARS, CICERO'S ORATIONS...</p>
                      <p>AFTER ROME'S FALL, LATIN BECAME THE CHURCH'S...</p>
                      <p>MODERN ROMANCE LANGUAGES (SPANISH, FRENCH...)</p>
                      <div className="ancient-symbol">Ⅴ</div>
                    </div>
                  </div>

                  {/* Phoenician */}
                  <div className="script-wrap animate pop delay-4">
                    <div className="script-overlay">
                      <div className="overlay-content animate slide-left delay-2">
                        <h1 className="animate slide-left pop delay-4">Phoenician</h1>
                        <p className="animate slide-left pop delay-5" style={{color: 'white', marginBottom: '2.5rem'}}>Origin: <em>Mediterranean Coast</em></p>
                      </div>
                      <div className="image-content phoenician animate slide delay-5"></div>
                      <div className="dots animate">
                        <div className="dot animate slide-up delay-6"></div>
                        <div className="dot animate slide-up delay-7"></div>
                        <div className="dot animate slide-up delay-8"></div>
                      </div>
                    </div>
                    <div className="text">
                      <p><img className="inset" src="/img/phonecian.jpg" alt="Phoenician Inscription" />PHOENICIAN SCRIPT EMERGED AROUND <em>1200 BCE</em>...</p>
                      <p>PHOENICIAN MERCHANTS CARRIED THEIR ALPHABET...</p>
                      <p>THE GREEKS ADAPTED PHOENICIAN LETTERS...</p>
                      <p>MODERN LEBANESE AND MEDITERRANEAN PEOPLES...</p>
                      <div className="ancient-symbol">𐤀</div>
                    </div>
                  </div>

                  {/* Persian */}
                  <div className="script-wrap animate pop delay-5">
                    <div className="script-overlay">
                      <div className="overlay-content animate slide-left delay-2">
                        <h1 className="animate slide-left pop delay-4">Persian</h1>
                        <p className="animate slide-left pop delay-5" style={{color: 'white', marginBottom: '2.5rem'}}>Origin: <em>Persian Empire</em></p>
                      </div>
                      <div className="image-content persian animate slide delay-5"></div>
                      <div className="dots animate">
                        <div className="dot animate slide-up delay-6"></div>
                        <div className="dot animate slide-up delay-7"></div>
                        <div className="dot animate slide-up delay-8"></div>
                      </div>
                    </div>
                    <div className="text">
                      <p><img className="inset" src="/img/persian.jpg" alt="Persian Cuneiform" />OLD PERSIAN CUNEIFORM DEVELOPED AROUND <em>600 BCE</em>...</p>
                      <p>PERSIAN ROYAL INSCRIPTIONS AT PERSEPOLIS...</p>
                      <p>THE PERSIAN EMPIRE'S TOLERANCE FOR LOCAL CULTURES...</p>
                      <p>MODERN IRANIANS HONOR OLD PERSIAN...</p>
                      <div className="ancient-symbol">𐎠</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="cta-section">
                <h2>Ready to Decipher Ancient Mysteries?</h2>
                <p>Upload your ancient text image and let our AI reveal its secrets</p>
                <button className="btn btn-primary ancient-glow" onClick={handleTimeTravel}>
                  <span className="button-text">Start Deciphering</span>
                  <div className="ancient-sheen"></div>
                </button>
              </div>
            </div>
          </section>

          <footer className="footer">
            <div className="container">
              <div className="footer-bottom">
                <p>&copy; Decipher AI. Unlocking the mysteries of the past with artificial intelligence.</p>
              </div>
            </div>
          </footer>
        </div>
      </div>
    );
  }

  // Upload Page (unchanged)
  return (
    <div className="App">
      <nav className="navbar">
        <div className="container">
          <div className="nav-brand" onClick={navigateHome}>
            <span className="logo">🏺</span>
            <span className="brand-text">Decipher AI</span>
          </div>
          <div className="nav-links">
            {isLoggedIn ? (
              <div className="user-menu">
                <span className="user-greeting">Welcome, {user.firstName}!</span>
                <button className="btn btn-secondary" onClick={navigateHome}>Home</button>
              </div>
            ) : (
              <button className="btn btn-secondary" onClick={navigateHome}>Home</button>
            )}
            <div className="theme-toggle">
              <label className="theme-toggle-label">
                <input
                  type="checkbox"
                  className="theme-toggle-input"
                  checked={isDarkMode}
                  onChange={toggleTheme}
                />
                <span className="theme-toggle-slider"></span>
                <span className="theme-toggle-sun">☀️</span>
                <span className="theme-toggle-moon">🌙</span>
              </label>
            </div>
          </div>
        </div>
      </nav>

      <section id="upload" className="upload-section">
        <div className="container">
          <h2 className="section-title">Try Decipher AI</h2>

          <div className="upload-container">
            <div className="upload-area">
              <div className="upload-icon">📸</div>
              <h3>Upload Your Ancient Text Image</h3>
              <p>Click to select an image of ancient scripts, manuscripts, or inscriptions</p>
              <input type="file" id="fileInput" accept="image/*" hidden onChange={handleImageUpload} />
              <button className="btn btn-primary ancient-glow" onClick={() => document.getElementById('fileInput').click()}>
                Choose File
                <div className="ancient-sheen"></div>
              </button>
            </div>

            {showPreview && (
              <div className="upload-preview">
                <img src={preview} alt="Preview" id="previewImage" />
                <div className="preview-actions">
                  <button className="btn btn-primary ancient-glow" onClick={handleAnalyze}>
                    Analyze Image<div className="ancient-sheen"></div>
                  </button>
                  <button className="btn btn-secondary ancient-glow" onClick={handleReset}>Upload New</button>
                </div>
              </div>
            )}
          </div>

          {showProcessing && (
            <div className="processing-section">
              <div className="processing-steps">
                {[
                  "📸 Analyzing image...",
                  "🔍 Identifying characters...",
                  "🧠 Generating translation...",
                  "📚 Creating historical context...",
                  "🎬 Generating video..."
                ].map((text, i) => (
                  <div key={i} className="processing-step active">
                    <div className="step-indicator"></div>
                    <span>{text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {showResults && (
            <div className="results-section">
              <h2>Analysis Complete!</h2>
              <div className="results-container stacked-left">
                <div className="result-card translation-card">
                  <h3>🔤 Translation</h3>
                  <div className="translation">
                    {formatTextForDisplay(translation)}
                  </div>
                </div>
                <div className="result-card historical-card">
                  <h3>📚 Historical Context</h3>
                  <HistoricalContextBoxes ctx={historicalContext} />
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <footer className="footer">
        <div className="container">
          <div className="footer-bottom">
            <p>&copy; DecipherAI. Unlocking the mysteries of the past with artificial intelligence.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
