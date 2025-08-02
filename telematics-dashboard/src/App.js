import React, { useState, useEffect, useRef } from 'react';
import { Smartphone, BarChart2, CloudRain, TrafficCone, Map, Shield, ShieldOff, FileText, Github } from 'lucide-react';

// --- Helper Components ---

const StatCard = ({ icon, title, value, color }) => (
  <div className={`bg-slate-800 p-4 rounded-lg shadow-md flex flex-col items-center justify-center text-center`}>
    <div className={`mb-2 text-${color}-400`}>{icon}</div>
    <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
    <p className="text-xl font-bold text-white">{value}</p>
  </div>
);

const Section = ({ title, icon, children }) => (
    <div className="bg-slate-800/50 rounded-xl shadow-lg p-6 border border-slate-700">
        <div className="flex items-center mb-4">
            <div className="text-cyan-400 mr-3">{icon}</div>
            <h2 className="text-xl font-bold text-white">{title}</h2>
        </div>
        {children}
    </div>
);

const Button = ({ onClick, children, disabled = false, className = '' }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`w-full px-4 py-3 font-bold text-white rounded-lg transition-all duration-300 flex items-center justify-center
            ${disabled 
                ? 'bg-slate-600 cursor-not-allowed' 
                : 'bg-cyan-600 hover:bg-cyan-500 shadow-lg hover:shadow-cyan-500/50'
            } ${className}`}
    >
        {children}
    </button>
);


// --- Main App Component ---

const App = () => {
    const [isCollecting, setIsCollecting] = useState(false);
    const [telemetryData, setTelemetryData] = useState([]);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Contextual settings
    const [weather, setWeather] = useState('clear');
    const [traffic, setTraffic] = useState('light');
    const [route, setRoute] = useState('safe');

    const collectionInterval = useRef(null);

    // --- Data Collection Logic ---
    const handleCollection = () => {
        if (!('DeviceMotionEvent' in window)) {
            alert("Sorry, your browser doesn't support device motion events. Please try on a mobile device.");
            return;
        }

        if (isCollecting) {
            // Stop collecting
            window.removeEventListener('devicemotion', motionHandler);
            setIsCollecting(false);
        } else {
            // Start collecting
            setTelemetryData([]);
            setAnalysisResult(null);
            // Request permission for iOS 13+
            if (typeof DeviceMotionEvent.requestPermission === 'function') {
                DeviceMotionEvent.requestPermission()
                    .then(permissionState => {
                        if (permissionState === 'granted') {
                            window.addEventListener('devicemotion', motionHandler);
                        }
                    })
                    .catch(console.error);
            } else {
                // For non-iOS 13+ devices
                window.addEventListener('devicemotion', motionHandler);
            }
            setIsCollecting(true);
        }
    };
    
    const motionHandler = (event) => {
        const { acceleration, rotationRate } = event;
        const newPoint = {
            accX: acceleration.x,
            accY: acceleration.y,
            accZ: acceleration.z,
            gyroX: rotationRate.alpha, // Corresponds to yaw
            gyroY: rotationRate.beta,  // Corresponds to pitch
            gyroZ: rotationRate.gamma, // Corresponds to roll
        };
        setTelemetryData(prevData => [...prevData, newPoint]);
    };
    
    useEffect(() => {
        if (isCollecting) {
            const timer = setTimeout(() => {
                setIsCollecting(false);
                window.removeEventListener('devicemotion', motionHandler);
                console.log("Collection stopped automatically after 15 seconds.");
            }, 15000); // Auto-stop after 15 seconds
            return () => clearTimeout(timer);
        }
    }, [isCollecting]);


    // --- Analysis Logic ---
    const handleAnalysis = () => {
        setIsAnalyzing(true);
        
        // Simulate a delay for processing
        setTimeout(() => {
            // 1. Simulate Behavior Score from Model
            let behaviorPrediction = 'SAFE';
            let behaviorScore = 1.0;
            let confidence = Math.random() * (0.98 - 0.85) + 0.85; // High confidence for SAFE
            
            if (telemetryData.length > 20) {
                const variance = telemetryData.reduce((acc, p) => acc + p.accX**2 + p.accY**2, 0) / telemetryData.length;
                // If variance is high, simulate an AGGRESSIVE event
                if (variance > 15) { 
                    behaviorPrediction = 'AGGRESSIVE';
                    behaviorScore = 1.5;
                    confidence = Math.random() * (0.95 - 0.80) + 0.80;
                }
            }

            // 2. Calculate Context Score
            const weatherMultipliers = { clear: 1.0, rain: 1.2, fog: 1.3, snow: 1.4 };
            const trafficMultipliers = { light: 1.0, moderate: 1.1, heavy: 1.25 };
            const weatherMultiplier = weatherMultipliers[weather];
            const trafficMultiplier = trafficMultipliers[traffic];
            const contextScore = (weatherMultiplier + trafficMultiplier) / 2;

            // 3. Calculate Route Score
            const routeMultipliers = { safe: 1.0, mixed: 1.1, hotspot: 1.25 };
            const routeScore = routeMultipliers[route];
            
            // 4. Final Premium Calculation
            const basePremium = 120.00;
            const w_behavior = 0.6;
            const w_context = 0.2;
            const w_route = 0.2;

            const finalRiskMultiplier = (w_behavior * behaviorScore) + (w_context * contextScore) + (w_route * routeScore);
            const finalPremium = basePremium * finalRiskMultiplier;

            setAnalysisResult({
                behaviorPrediction,
                behaviorScore,
                confidence,
                contextScore,
                routeScore,
                finalRiskMultiplier,
                basePremium,
                finalPremium,
                weather,
                traffic,
                route,
            });
            setIsAnalyzing(false);
        }, 2500);
    };
    
    const ContextSelector = ({ value, onChange, options, icon }) => (
        <div className="relative">
            <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-slate-400">
                {icon}
            </div>
            <select
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 text-white text-sm rounded-lg focus:ring-cyan-500 focus:border-cyan-500 block pl-10 p-2.5"
            >
                {Object.entries(options).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                ))}
            </select>
        </div>
    );

    return (
        <div className="min-h-screen bg-slate-900 text-white font-sans p-4 sm:p-6 lg:p-8">
            <div className="max-w-4xl mx-auto">
                {/* --- Header --- */}
                <header className="text-center mb-8">
                    <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-500">
                        Context-Aware Telematics Dashboard
                    </h1>
                    <p className="text-slate-400 mt-2">
                        A Proof-of-Concept for Dynamic, Usage-Based Insurance.
                    </p>
                </header>

                <main className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* --- Left Column: Collection & Configuration --- */}
                    <div className="flex flex-col gap-8">
                        <Section title="Data Collection" icon={<Smartphone size={24} />}>
                            <p className="text-slate-400 mb-4 text-sm">
                                {isCollecting 
                                    ? "Move your phone to simulate a driving trip. Collection will stop automatically."
                                    : "Click the button below to start collecting 15 seconds of motion data from your phone's sensors."
                                }
                            </p>
                            <Button onClick={handleCollection} className={isCollecting ? 'bg-red-600 hover:bg-red-500' : ''}>
                                {isCollecting ? (
                                    <>
                                        <div className="animate-pulse mr-2">🔴</div>
                                        Stop Collection ({telemetryData.length})
                                    </>
                                ) : (
                                    <>
                                        <Smartphone className="mr-2" size={20} />
                                        {telemetryData.length > 0 ? 'Start New Collection' : 'Start Collection'}
                                    </>
                                )}
                            </Button>
                        </Section>

                        <Section title="Context Simulation" icon={<BarChart2 size={24} />}>
                             <p className="text-slate-400 mb-4 text-sm">
                                Select the environmental and route conditions for the trip.
                            </p>
                            <div className="space-y-4">
                               <ContextSelector 
                                   value={weather}
                                   onChange={setWeather}
                                   options={{clear: 'Clear Day', rain: 'Rainy', fog: 'Foggy', snow: 'Snowy'}}
                                   icon={<CloudRain size={16} />}
                               />
                               <ContextSelector 
                                   value={traffic}
                                   onChange={setTraffic}
                                   options={{light: 'Light Traffic', moderate: 'Moderate Traffic', heavy: 'Heavy Traffic'}}
                                   icon={<TrafficCone size={16} />}
                               />
                               <ContextSelector 
                                   value={route}
                                   onChange={setRoute}
                                   options={{safe: 'Safe Residential Route', mixed: 'Mixed Urban Route', hotspot: 'Accident Hotspot Route'}}
                                   icon={<Map size={16} />}
                               />
                            </div>
                        </Section>
                        
                        <Button onClick={handleAnalysis} disabled={telemetryData.length < 20 || isAnalyzing}>
                            {isAnalyzing ? "Analyzing..." : "Calculate Premium"}
                        </Button>
                    </div>

                    {/* --- Right Column: Results --- */}
                    <div>
                        <Section title="Analysis Report" icon={<FileText size={24} />}>
                            {isAnalyzing && (
                                <div className="text-center py-10">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto"></div>
                                    <p className="mt-4 text-slate-300">Running analysis through risk engine...</p>
                                </div>
                            )}
                            {!isAnalyzing && !analysisResult && (
                                <div className="text-center py-10 text-slate-400">
                                    <p>Your dynamic premium report will appear here after analysis.</p>
                                </div>
                            )}
                            {analysisResult && (
                                <div className="space-y-6 animate-fade-in">
                                    {/* Behavior Result */}
                                    <div className={`p-4 rounded-lg text-center ${analysisResult.behaviorPrediction === 'AGGRESSIVE' ? 'bg-red-500/20' : 'bg-green-500/20'}`}>
                                        <div className={`mx-auto w-fit mb-2 ${analysisResult.behaviorPrediction === 'AGGRESSIVE' ? 'text-red-400' : 'text-green-400'}`}>
                                            {analysisResult.behaviorPrediction === 'AGGRESSIVE' ? <ShieldOff size={32}/> : <Shield size={32}/>}
                                        </div>
                                        <h3 className="text-lg font-bold text-white">
                                            Driving Style: {analysisResult.behaviorPrediction}
                                        </h3>
                                        <p className="text-sm text-slate-300">
                                            Model Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
                                        </p>
                                    </div>
                                    
                                    {/* Score Breakdown */}
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                                        <StatCard icon={<Shield size={24}/>} title="Behavior Score" value={analysisResult.behaviorScore.toFixed(2) + 'x'} color="cyan" />
                                        <StatCard icon={<BarChart2 size={24}/>} title="Context Score" value={analysisResult.contextScore.toFixed(2) + 'x'} color="yellow" />
                                        <StatCard icon={<Map size={24}/>} title="Route Score" value={analysisResult.routeScore.toFixed(2) + 'x'} color="orange" />
                                    </div>

                                    {/* Final Premium */}
                                    <div className="text-center bg-slate-900/50 p-6 rounded-lg">
                                        <p className="text-slate-400 text-sm">Base Premium</p>
                                        <p className="text-2xl font-light text-slate-300 line-through">${analysisResult.basePremium.toFixed(2)}</p>
                                        <p className="text-cyan-400 text-sm mt-4">Final Dynamic Premium</p>
                                        <p className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-500">
                                            ${analysisResult.finalPremium.toFixed(2)}
                                        </p>
                                        <p className="text-xs text-slate-500 mt-2">
                                            Based on a final risk multiplier of {analysisResult.finalRiskMultiplier.toFixed(2)}x
                                        </p>
                                    </div>
                                </div>
                            )}
                        </Section>
                    </div>
                </main>
                <footer className="text-center mt-8 text-slate-500 text-xs">
                    <p>Proof of Concept for Insurity Telematics Integration.</p>
                     <a href="https://github.com/your-repo-link" target="_blank" rel="noopener noreferrer" className="inline-flex items-center hover:text-cyan-400 transition-colors">
                        <Github size={14} className="mr-1"/> View on GitHub
                    </a>
                </footer>
            </div>
        </div>
    );
};

export default App;
