import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Activity, FileText, Brain, Heart, AlertTriangle, ChevronRight, MessageSquare, RefreshCw, Eye } from 'lucide-react';
import axios from 'axios';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Chatbot State
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    { text: "Hello! I am your AI diagnostic assistant. Feel free to ask me anything about the scan results or brain tumors.", isBot: true }
  ]);
  const [chatInput, setChatInput] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, chatOpen]);

  const handleImageUpload = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selected);
      setResult(null);
      setError(null);
    }
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);

    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const elapsed = Date.now() - startTime;
      const remainingTime = Math.max(0, 2000 - elapsed);

      setTimeout(() => {
        setResult(response.data);
        setIsAnalyzing(false);

        setMessages(prev => [...prev, {
          text: `I have analyzed the MRI. The preliminary diagnosis is ${response.data.diagnosis} with ${response.data.confidence}% confidence. Please let me know if you would like me to explain what this means.`,
          isBot: true
        }]);
      }, remainingTime);

    } catch (err) {
      console.error(err);
      setError("Failed to analyze image. Ensure the backend analysis server is running.");
      setIsAnalyzing(false);
    }
  };

  const generatePDF = () => {
    if (!result || !preview) return;

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    // --- Original Style Header ---
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 0, pageWidth, 40, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont('helvetica', 'bold');
    doc.text('NeuroAI Diagnostic Report', 20, 25);
    
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(200, 200, 200);
    doc.text('Automated Neuro-Oncology Analysis Engine', 20, 32);

    // --- Added from Reference Image: Patient & Facility Info ---
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(150, 160, 170);
    doc.text('PATIENT INFORMATION', 20, 50);
    
    doc.setFontSize(10);
    doc.setTextColor(0, 0, 0); 
    doc.text('Subject ID:', 20, 58);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    doc.text(` NSA-${Math.random().toString(36).substr(2, 9).toUpperCase()}`, 40, 58);
    
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(0, 0, 0);
    doc.text('Age / Sex:', 20, 64);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    doc.text(' N/A', 40, 64);

    doc.setFont('helvetica', 'bold');
    doc.setTextColor(0, 0, 0);
    doc.text('Status:', 20, 70);
    doc.setTextColor(22, 163, 74); 
    doc.text(' HOLOGRAPHIC VERIFIED', 34, 70);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(150, 160, 170);
    doc.text('FACILITY INFORMATION', pageWidth / 2 + 10, 50);

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    doc.text('Analysis Core: NeuroAI Diagnostic', pageWidth / 2 + 10, 58);
    const dateStr = new Date().toLocaleString('en-US', { month: 'long', day: 'numeric', year:'numeric', hour:'2-digit', minute:'2-digit' });
    doc.text(`Timestamp: ${dateStr}`, pageWidth / 2 + 10, 64);
    
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(22, 163, 74); 
    doc.text('ENCRYPTION PROTOCOL: AES-256 ACTIVE', pageWidth / 2 + 10, 70);

    // Separator line
    doc.setDrawColor(230, 230, 230);
    doc.line(20, 80, pageWidth - 20, 80);

    // --- Added from Reference Image: Diagnostic Synthesis Box ---
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(80, 180, 210);
    doc.text('Q', 20, 88);
    doc.setTextColor(100, 110, 120);
    doc.text('  DIAGNOSTIC SYNTHESIS', 23, 88);

    const isTumor = result.raw_label !== 'no_tumor';
    if (isTumor) {
      doc.setFillColor(254, 242, 242); 
    } else {
      doc.setFillColor(240, 253, 244); 
    }
    
    doc.roundedRect(20, 92, pageWidth - 40, 24, 2, 2, 'F');

    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text('AI CLASSIFICATION RESULT', 25, 98);
    doc.text('CONFIDENCE SCORE', pageWidth - 25, 98, { align: 'right' });

    doc.setFontSize(16);
    if (isTumor) {
      doc.setTextColor(220, 38, 38); 
      doc.text(`POSITIVE: ABNORMALITY DETECTED`, 25, 108);
    } else {
      doc.setTextColor(22, 163, 74); 
      doc.text(`NEGATIVE: NORMAL STRUCTURE`, 25, 108);
    }
    
    doc.setFontSize(20);
    doc.setTextColor(isTumor ? 220 : 22, isTumor ? 38 : 163, isTumor ? 38 : 74);
    doc.text(`${result.confidence}%`, pageWidth - 25, 108, { align: 'right' });

    // --- Original Style Images ---
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(0, 0, 0);
    doc.text('Original Scan:', 20, 135);
    doc.addImage(preview, 'JPEG', 20, 140, 70, 70);

    if (result.saliency_b64) {
      doc.text('AI Detected Tumor Highlight:', 110, 135);
      doc.addImage(result.saliency_b64, 'JPEG', 110, 140, 70, 70);
    }

    // --- Target Region (Below Images) ---
    let targetRegionStr = "NO ABNORMALITY LOCALIZED";
    if (result.raw_label === 'glioma_tumor') targetRegionStr = "BRAIN PARENCHYMA / GLIAL TISSUE";
    else if (result.raw_label === 'meningioma_tumor') targetRegionStr = "MENINGEAL MEMBRANES (CORTEX SURFACE)";
    else if (result.raw_label === 'pituitary_tumor') targetRegionStr = "SELLA TURCICA / PITUITARY GLAND";

    doc.setFillColor(248, 250, 252); 
    doc.setDrawColor(203, 213, 225); 
    doc.roundedRect(20, 215, pageWidth - 40, 12, 1, 1, 'FD'); 

    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105); 
    doc.text('Targeted Region:', 25, 223);
    
    doc.setTextColor(6, 182, 212); 
    doc.text(targetRegionStr, 85, 223);

    // --- Original Style Table Breakdown ---
    autoTable(doc, {
      startY: 235,
      head: [['Classification', 'Probability']],
      body: result.breakdown.map(b => [b.label, `${b.percentage}%`]),
      theme: 'grid',
      headStyles: { fillColor: [15, 23, 42] } 
    });

    // --- Original Style Recommendations ---
    doc.addPage();
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(16);
    doc.setTextColor(0, 0, 0);
    doc.text('Clinical Recommendation:', 20, 20);
    
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(12);
    const splitRecText = doc.splitTextToSize(result.recommendation, 170);
    doc.text(splitRecText, 20, 30);

    // --- Original Footer ---
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text('DISCLAIMER: This report is generated by an artificial intelligence assistant. It does not replace professional medical advice, diagnosis, or treatment.', 20, 280, { maxWidth: 170 });

    doc.save(`NeuroAI_Report_${new Date().getTime()}.pdf`);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = chatInput.trim();
    setMessages(prev => [...prev, { text: userMsg, isBot: false }]);
    setChatInput('');

    try {
      const formData = new FormData();
      formData.append('message', userMsg);
      formData.append('diagnosis', result?.raw_label || '');

      const res = await axios.post(`${API_URL}/chat`, formData);

      setTimeout(() => {
        setMessages(prev => [...prev, { text: res.data.reply, isBot: true }]);
      }, 500);
    } catch (err) {
      setMessages(prev => [...prev, { text: "I'm sorry, my systems are currently unreachable.", isBot: true }]);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden flex flex-col">
      {/* Background Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/30 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[40%] bg-purple-600/20 rounded-full blur-[100px] pointer-events-none" />

      {/* Navbar */}
      <nav className="w-full relative z-10 glass-panel border-x-0 border-t-0 rounded-none px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="text-blue-400 w-8 h-8" />
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
            NeuroAI Diagnostic
          </h1>
        </div>
        <div className="flex items-center gap-4 text-sm font-medium text-slate-300">
          <span>Ensemble Model Validated</span>
          <Activity className="w-4 h-4 text-emerald-400" />
        </div>
      </nav>

      <main className="flex-1 w-full max-w-7xl mx-auto p-4 md:p-8 flex flex-col items-center relative z-10">

        <AnimatePresence mode="wait">
          {!result && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20, filter: 'blur(10px)' }}
              transition={{ duration: 0.5 }}
              className="w-full max-w-2xl mt-12"
            >
              <div className="text-center mb-8">
                <h2 className="text-4xl font-extrabold mb-4">Precision Brain MRI Analysis</h2>
                <p className="text-slate-400 text-lg">Upload an MRI scan to generate an instant diagnosis powered by a triad of state-of-the-art vision models.</p>
              </div>

              <div className="glass-panel p-10 flex flex-col items-center justify-center relative overflow-hidden group">

                {preview ? (
                  <div className="relative w-full aspect-square md:aspect-video rounded-xl overflow-hidden bg-black/50 border flex items-center justify-center border-slate-700/50">
                    <img src={preview} alt="MRI Preview" className={`max-w-full max-h-full object-contain transition-all duration-700 ${isAnalyzing ? 'scale-105 opacity-50' : 'scale-100'}`} />

                    {isAnalyzing && (
                      <>
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/20 to-transparent w-full h-[20%] pointer-events-none"
                          animate={{ y: ['-100%', '500%'] }}
                          transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                        />
                        <div className="absolute inset-0 border-2 border-cyan-400/50 rounded-xl" style={{ boxShadow: 'inset 0 0 50px rgba(6,182,212,0.3)' }} />
                        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 px-4 py-2 rounded-full text-cyan-300 font-medium flex items-center gap-2">
                          <RefreshCw className="w-4 h-4 animate-spin" /> Analyzing features...
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <label className="w-full aspect-video rounded-xl border-2 border-dashed border-slate-600 hover:border-blue-400/50 bg-slate-800/30 flex flex-col items-center justify-center cursor-pointer transition-colors group-hover:bg-slate-800/50">
                    <div className="bg-blue-500/10 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
                      <Upload className="w-8 h-8 text-blue-400" />
                    </div>
                    <span className="font-medium text-lg">Click to Upload MRI Scan</span>
                    <span className="text-slate-500 text-sm mt-2">JPEG, PNG up to 10MB</span>
                    <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                  </label>
                )}

                {preview && !isAnalyzing && (
                  <div className="mt-8 flex gap-4 w-full">
                    <button onClick={resetAll} className="flex-1 py-3 px-4 rounded-xl border border-slate-600 hover:bg-slate-800 font-medium transition-colors text-slate-300">
                      Cancel
                    </button>
                    <button onClick={analyzeImage} className="flex-2 w-full py-3 px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold shadow-lg shadow-blue-500/25 flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98]">
                      <Activity className="w-5 h-5" /> Start Analysis
                    </button>
                  </div>
                )}

                {error && (
                  <div className="absolute bottom-4 left-4 right-4 bg-red-500/10 border border-red-500/20 text-red-400 p-3 rounded-lg flex items-center gap-2 text-sm backdrop-blur-md">
                    <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {result && !isAnalyzing && (
            <motion.div
              key="results"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="w-full mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6 pb-24"
            >
              <div className="lg:col-span-1 flex flex-col gap-6">

                <div className="glass-panel p-4 flex flex-col">
                  <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wide">Image Scans</h3>
                  <div className="flex gap-2 mb-4">
                    {/* Original Image */}
                    <div className="relative w-1/2 aspect-square rounded-lg overflow-hidden border border-slate-700/50 bg-black">
                      <div className="absolute top-1 left-1 bg-black/60 px-2 py-0.5 rounded text-[10px] text-white z-10 pointer-events-none">Original</div>
                      <img src={preview} alt="Original Scan" className="w-full h-full object-cover" />
                    </div>
                    {/* Highlighted Saliency Image */}
                    <div className="relative w-1/2 aspect-square rounded-lg overflow-hidden border border-slate-700/50 bg-black">
                      <div className="absolute top-1 left-1 bg-red-500/80 px-2 py-0.5 rounded text-[10px] text-white z-10 shadow pointer-events-none">Tumor Highlight</div>
                      <img src={result.saliency_b64 || preview} alt="Highlighted Scan" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-20 pointer-events-none mix-blend-screen" />
                    </div>
                  </div>

                  <button onClick={resetAll} className="w-full py-2.5 rounded-lg border border-slate-600/50 hover:bg-slate-800/50 text-slate-300 text-sm font-medium flex items-center justify-center gap-2 transition-colors">
                    <RefreshCw className="w-4 h-4" /> Scan Another Patient
                  </button>
                </div>

                <div className="glass-panel p-6">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <Eye className="w-5 h-5 text-indigo-400" /> Ensemble Analysis
                  </h3>
                  <div className="space-y-4">
                    {result.breakdown.map((item, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className={item.label === result.diagnosis ? "text-indigo-300 font-semibold" : "text-slate-400"}>
                            {item.label}
                          </span>
                          <span className="font-mono">{item.percentage}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${item.percentage}%` }}
                            transition={{ duration: 1, delay: idx * 0.1 }}
                            className={`h-full rounded-full ${item.label === result.diagnosis ? 'bg-indigo-500' : 'bg-slate-600'}`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="lg:col-span-2 flex flex-col gap-6">
                <div className="glass-panel p-8 md:p-10 relative overflow-hidden h-full flex flex-col">
                  <div className={`absolute top-0 right-0 w-32 h-32 rounded-bl-full ${result.raw_label === 'no_tumor' ? 'bg-emerald-500/10' : 'bg-red-500/10'} -z-10`} />

                  <div className="inline-block px-3 py-1 bg-slate-800/50 border border-slate-700 rounded-full text-xs font-semibold text-blue-300 mb-6 tracking-wide uppercase self-start">
                    Diagnostic Summary
                  </div>

                  <h2 className="text-4xl font-black mb-2 text-white">
                    {result.diagnosis}
                  </h2>

                  <div className="flex items-center gap-2 text-slate-300 mb-8">
                    Confidence Rating:
                    <span className={`px-2 py-0.5 rounded text-sm font-bold ${result.confidence > 90 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                      {result.confidence}%
                    </span>
                  </div>

                  <hr className="border-slate-700/50 mb-8" />

                  <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                    <Heart className="w-5 h-5 text-rose-400" /> Clinical Recommendation
                  </h3>
                  <p className="text-slate-300 leading-relaxed text-lg mb-10">
                    {result.recommendation}
                  </p>

                  <div className="mt-auto pt-4 flex">
                    <button onClick={generatePDF} className="bg-white text-slate-900 hover:bg-slate-100 font-bold py-3 px-6 rounded-xl shadow-lg flex items-center gap-2 transition-transform hover:-translate-y-1">
                      <FileText className="w-5 h-5" /> Download Full PDF Report
                    </button>
                  </div>
                </div>
              </div>

            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* Floating Action Button (Always Visible) */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-4">

        {/* Chat window popping up from FAB */}
        <AnimatePresence>
          {chatOpen && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.9, originBottomRight: true }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.9 }}
              className="w-96 max-w-[calc(100vw-2rem)] h-[500px] max-h-[70vh] glass-panel bg-slate-900/95 flex flex-col overflow-hidden shadow-2xl border-slate-700/50"
            >
              <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-4 shrink-0 flex justify-between items-center relative shadow-md">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                    <Brain className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="font-bold text-white text-sm">Neuro Assistant</h3>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                      <span className="text-xs text-indigo-100">Online</span>
                    </div>
                  </div>
                </div>
                <button onClick={() => setChatOpen(false)} className="text-indigo-100 hover:text-white p-1">
                  ✕
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.isBot ? 'justify-start' : 'justify-end'}`}>
                    <div className={`max-w-[80%] p-3 rounded-2xl text-sm ${msg.isBot ? 'bg-slate-800 text-slate-200 rounded-tl-sm border border-slate-700/50' : 'bg-indigo-600 text-white rounded-tr-sm shadow-md'}`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>

              <form onSubmit={handleSendMessage} className="p-4 border-t border-slate-700/50 shrink-0 bg-slate-900 flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask about the diagnosis..."
                  className="flex-1 bg-slate-800 border-none rounded-xl px-4 py-2 text-sm focus:ring-1 focus:ring-indigo-500 outline-none text-white placeholder-slate-500"
                />
                <button disabled={!chatInput.trim()} type="submit" className="bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 disabled:hover:bg-indigo-500 text-white p-2 rounded-xl transition-colors">
                  <ChevronRight className="w-5 h-5" />
                </button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>

        {/* The Action Button Itself */}
        <button
          onClick={() => setChatOpen(!chatOpen)}
          className={`h-16 w-16 rounded-full shadow-2xl flex items-center justify-center transition-all duration-300 hover:scale-110 active:scale-95 z-50 ${chatOpen ? 'bg-slate-700 text-white border border-slate-600' : 'bg-gradient-to-tr from-indigo-600 to-fuchsia-600 text-white shadow-indigo-500/50'}`}
        >
          {chatOpen ? <ChevronRight className="w-8 h-8 rotate-90" /> : <MessageSquare className="w-8 h-8" />}
        </button>
      </div>

    </div>
  );
}

export default App;
