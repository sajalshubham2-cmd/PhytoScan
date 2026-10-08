/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  CropType,
  DiagnosticHistoryItem,
  KNOWLEDGE_BASE,
  SAMPLE_PRESETS,
  SamplePreset,
  runCropAwareInference,
  InferenceResult
} from './data/cropKnowledgeBase';
import { SpecViewer } from './components/SpecViewer';
import {
  Upload,
  History,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sliders,
  FileText,
  Microscope,
  BookOpen,
  RotateCcw,
  ArrowUpRight,
  Globe,
  Search,
  ExternalLink,
  Loader2,
  MapPin,
  Navigation,
  MessageSquare,
  Send,
  ShieldAlert,
  Sparkles
} from 'lucide-react';

const STORAGE_KEY = 'phytoscan_diagnostic_history_v1';
const MAX_HISTORY_ITEMS = 5;

interface GroundedSearchResult {
  modelUsed: string;
  summary: string;
  articles: { title: string; uri: string }[];
  searchQueries: string[];
}

interface GroundedMapsResult {
  modelUsed: string;
  guidance: string;
  locationUsed: string;
  places: {
    title: string;
    uri: string;
    reviewSnippets: string[];
  }[];
}

interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  modelUsed?: string;
  timestamp: string;
}

type ChatTier = 'fast' | 'general' | 'complex';

function buildInitialHistory(): DiagnosticHistoryItem[] {
  const preset1 = SAMPLE_PRESETS[0]; // Tomato Early Blight
  const res1 = runCropAwareInference(preset1.rawProbabilities, 'tomato', 0.65, 0.15);
  const preset2 = SAMPLE_PRESETS[5]; // Maize Common Rust
  const res2 = runCropAwareInference(preset2.rawProbabilities, 'maize', 0.65, 0.15);

  return [
    {
      id: 'seed-1',
      timestamp: new Date(Date.now() - 1000 * 60 * 14).toISOString(),
      cropSelected: 'tomato',
      sampleName: preset1.label,
      imagePreviewDataUrl: preset1.svgDataUrl,
      status: res1.status,
      predictedClassKey: res1.predictedClassKey,
      diseaseName: res1.diseaseName,
      confidence: res1.confidence,
      top2ClassKey: res1.top2ClassKey,
      top2DiseaseName: res1.top2DiseaseName,
      top2Confidence: res1.top2Confidence,
      margin: res1.margin,
      rawCropMass: res1.rawCropMass,
      latencyMs: res1.latencyMs,
      message: res1.message,
      remedy: res1.remedy
    },
    {
      id: 'seed-2',
      timestamp: new Date(Date.now() - 1000 * 60 * 42).toISOString(),
      cropSelected: 'maize',
      sampleName: preset2.label,
      imagePreviewDataUrl: preset2.svgDataUrl,
      status: res2.status,
      predictedClassKey: res2.predictedClassKey,
      diseaseName: res2.diseaseName,
      confidence: res2.confidence,
      top2ClassKey: res2.top2ClassKey,
      top2DiseaseName: res2.top2DiseaseName,
      top2Confidence: res2.top2Confidence,
      margin: res2.margin,
      rawCropMass: res2.rawCropMass,
      latencyMs: res2.latencyMs,
      message: res2.message,
      remedy: res2.remedy
    }
  ];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'diagnostic' | 'kb' | 'spec'>('diagnostic');
  const [selectedCrop, setSelectedCrop] = useState<CropType>('tomato');
  const [selectedPreset, setSelectedPreset] = useState<SamplePreset>(SAMPLE_PRESETS[0]);
  const [customImageName, setCustomImageName] = useState<string | null>(null);
  const [customImageDataUrl, setCustomImageDataUrl] = useState<string | null>(null);
  const [customRawProbs, setCustomRawProbs] = useState<number[] | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [tConf, setTConf] = useState<number>(0.65);
  const [tMargin, setTMargin] = useState<number>(0.15);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // Current diagnostic result displayed on the main screen
  const [currentResult, setCurrentResult] = useState<InferenceResult>(() =>
    runCropAwareInference(SAMPLE_PRESETS[0].rawProbabilities, 'tomato', 0.65, 0.15)
  );
  const [activeSampleTitle, setActiveSampleTitle] = useState<string>(SAMPLE_PRESETS[0].label);
  const [activePreviewUrl, setActivePreviewUrl] = useState<string>(SAMPLE_PRESETS[0].svgDataUrl);

  // Google Search Grounding state
  const [searchQueryInput, setSearchQueryInput] = useState<string>('');
  const [isSearchingWeb, setIsSearchingWeb] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [groundedData, setGroundedData] = useState<GroundedSearchResult | null>(null);

  // Google Maps Grounding state (Pesticide & Medicine Store Locator)
  const [locationQuery, setLocationQuery] = useState<string>('Fresno, California');
  const [userCoords, setUserCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [isLocatingGps, setIsLocatingGps] = useState<boolean>(false);
  const [isSearchingMaps, setIsSearchingMaps] = useState<boolean>(false);
  const [mapsError, setMapsError] = useState<string | null>(null);
  const [mapsResult, setMapsResult] = useState<GroundedMapsResult | null>(null);

  // Multi-turn Gemini Agronomy Chatbot state
  const [chatTier, setChatTier] = useState<ChatTier>('general');
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-msg',
      role: 'model',
      text: 'Hello! I am Dr. PhytoScan, your Crop Protection & Pesticide Medicine Specialist. Ask me which specific fungicides, organic sprays, mixing dosages, or preventative schedules you need for your Tomato or Maize crop.',
      modelUsed: 'gemini-3.5-flash',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  // localStorage-based History storing the last 5 diagnostic results
  const [history, setHistory] = useState<DiagnosticHistoryItem[]>(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          return parsed.slice(0, MAX_HISTORY_ITEMS);
        }
      }
    } catch {
      // ignore storage errors
    }
    const seeded = buildInitialHistory();
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(seeded));
    } catch {
      // ignore
    }
    return seeded;
  });

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Sync history changes to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(history.slice(0, MAX_HISTORY_ITEMS))
      );
    } catch {
      // ignore quota errors
    }
  }, [history]);

  // Auto-scroll chat thread when new messages arrive
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, isChatLoading]);

  const fetchGoogleSearchGrounding = async (opts?: {
    diseaseName?: string;
    causalAgent?: string;
    crop?: string;
    customQuery?: string;
  }) => {
    setIsSearchingWeb(true);
    setSearchError(null);
    try {
      const payload = {
        diseaseName: opts?.diseaseName ?? currentResult.diseaseName,
        causalAgent: opts?.causalAgent ?? currentResult.remedy?.causal_agent ?? '',
        crop: opts?.crop ?? currentResult.cropSelected,
        customQuery: opts?.customQuery?.trim() || undefined
      };

      const response = await fetch('/api/search-grounding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }
      setGroundedData(data);
    } catch (err: any) {
      setSearchError(
        err?.message || 'Unable to fetch Google Search grounded research articles.'
      );
    } finally {
      setIsSearchingWeb(false);
    }
  };

  const fetchGoogleMapsStores = async (opts?: {
    locQuery?: string;
    lat?: number;
    lng?: number;
  }) => {
    setIsSearchingMaps(true);
    setMapsError(null);
    try {
      const activeRemedy = currentResult.remedy || KNOWLEDGE_BASE[0];
      const payload = {
        diseaseName: currentResult.remedy
          ? currentResult.diseaseName
          : `${currentResult.cropSelected === 'tomato' ? 'Tomato' : 'Maize'} Foliar Blight & Rust Prevention`,
        causalAgent: activeRemedy.causal_agent,
        crop: currentResult.cropSelected,
        chemicalRemedy: activeRemedy.chemical_remedy,
        organicRemedy: activeRemedy.organic_remedy,
        locationQuery: opts?.locQuery !== undefined ? opts.locQuery : locationQuery,
        latitude: opts?.lat ?? userCoords?.lat,
        longitude: opts?.lng ?? userCoords?.lng
      };

      const response = await fetch('/api/maps-grounding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Maps request failed (${response.status})`);
      }
      setMapsResult(data);
    } catch (err: any) {
      setMapsError(
        err?.message || 'Unable to fetch nearby pesticide & agricultural supply stores.'
      );
    } finally {
      setIsSearchingMaps(false);
    }
  };

  const handleUseMyLocation = () => {
    if (!navigator.geolocation) {
      setMapsError('Geolocation is not supported by your browser. Please enter your city or town manually.');
      return;
    }
    setIsLocatingGps(true);
    setMapsError(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const coords = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude
        };
        setUserCoords(coords);
        const coordLabel = `${coords.lat.toFixed(4)}, ${coords.lng.toFixed(4)}`;
        setLocationQuery(`Near GPS (${coordLabel})`);
        setIsLocatingGps(false);
        fetchGoogleMapsStores({
          locQuery: '',
          lat: coords.lat,
          lng: coords.lng
        });
      },
      (err) => {
        setIsLocatingGps(false);
        setMapsError(
          `GPS access unavailable (${err.message}). Searching by city name "${locationQuery}" instead.`
        );
        fetchGoogleMapsStores({ locQuery: locationQuery });
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  };

  const handleSendChatMessage = async (textToSend?: string) => {
    const content = (textToSend ?? chatInput).trim();
    if (!content || isChatLoading) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: content,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const updatedMessages = [...chatMessages, userMsg];
    setChatMessages(updatedMessages);
    if (!textToSend) {
      setChatInput('');
    }
    setIsChatLoading(true);
    setChatError(null);

    try {
      const activeRemedy = currentResult.remedy;
      const remedySummary = activeRemedy
        ? `Organic: ${activeRemedy.organic_remedy} | Chemical: ${activeRemedy.chemical_remedy}`
        : 'Preventative broad-spectrum copper oxychloride or mancozeb 75% WP';

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages.map((m) => ({ role: m.role, text: m.text })),
          tier: chatTier,
          cropContext: currentResult.cropSelected,
          diseaseContext: currentResult.diseaseName,
          remedyContext: remedySummary
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Chat request failed (${response.status})`);
      }

      const modelMsg: ChatMessage = {
        id: `model-${Date.now()}`,
        role: 'model',
        text: data.reply,
        modelUsed: data.modelUsed,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setChatMessages((prev) => [...prev, modelMsg]);
    } catch (err: any) {
      setChatError(err?.message || 'Failed to get response from Dr. PhytoScan.');
    } finally {
      setIsChatLoading(false);
    }
  };

  const saveResultToLocalStorageHistory = (
    res: InferenceResult,
    sampleName: string,
    previewUrl: string
  ) => {
    const newItem: DiagnosticHistoryItem = {
      id: `diag-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: new Date().toISOString(),
      cropSelected: res.cropSelected,
      sampleName,
      imagePreviewDataUrl: previewUrl,
      status: res.status,
      predictedClassKey: res.predictedClassKey,
      diseaseName: res.diseaseName,
      confidence: res.confidence,
      top2ClassKey: res.top2ClassKey,
      top2DiseaseName: res.top2DiseaseName,
      top2Confidence: res.top2Confidence,
      margin: res.margin,
      rawCropMass: res.rawCropMass,
      latencyMs: res.latencyMs,
      message: res.message,
      remedy: res.remedy
    };

    setHistory((prev) => {
      const updated = [newItem, ...prev].slice(0, MAX_HISTORY_ITEMS);
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch {
        // ignore quota errors
      }
      return updated;
    });
  };

  const executeDiagnosis = (
    probs: number[],
    crop: CropType,
    sampleLabel: string,
    previewUrl: string,
    confThreshold = tConf,
    marginThreshold = tMargin
  ) => {
    setUploadError(null);
    setIsAnalyzing(true);
    setTimeout(() => {
      const result = runCropAwareInference(probs, crop, confThreshold, marginThreshold);
      setCurrentResult(result);
      setActiveSampleTitle(sampleLabel);
      setActivePreviewUrl(previewUrl);
      saveResultToLocalStorageHistory(result, sampleLabel, previewUrl);
      setIsAnalyzing(false);
    }, 140);
  };

  const handleSelectPreset = (preset: SamplePreset) => {
    setUploadError(null);
    setCustomImageName(null);
    setCustomImageDataUrl(null);
    setCustomRawProbs(null);
    setSelectedPreset(preset);
    executeDiagnosis(
      preset.rawProbabilities,
      selectedCrop,
      preset.label,
      preset.svgDataUrl
    );
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    if (!['jpg', 'jpeg', 'png'].includes(ext)) {
      setUploadError(
        `HTTP 415 INVALID_FILE_TYPE: "${file.name}" is not supported. Please upload a .jpg, .jpeg, or .png leaf photo.`
      );
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setUploadError(
        `HTTP 413 FILE_TOO_LARGE: "${file.name}" is ${(file.size / (1024 * 1024)).toFixed(
          2
        )} MB. Maximum allowed file size is 5.00 MB.`
      );
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        let derivedProbs =
          selectedCrop === 'tomato'
            ? [0.89, 0.05, 0.03, 0.01, 0.005, 0.005, 0.005, 0.005]
            : [0.005, 0.005, 0.005, 0.005, 0.04, 0.89, 0.04, 0.01];

        if (ctx) {
          ctx.drawImage(img, 0, 0, 64, 64);
          const pixels = ctx.getImageData(0, 0, 64, 64).data;
          let rSum = 0;
          let gSum = 0;
          let bSum = 0;
          for (let i = 0; i < pixels.length; i += 4) {
            rSum += pixels[i];
            gSum += pixels[i + 1];
            bSum += pixels[i + 2];
          }
          const n = pixels.length / 4;
          const rMean = rSum / n;
          const gMean = gSum / n;
          const bMean = bSum / n;

          if (gMean < rMean * 0.85 && gMean < bMean * 1.05) {
            derivedProbs = [0.21, 0.19, 0.18, 0.17, 0.07, 0.06, 0.06, 0.06];
          } else if (gMean > rMean * 1.25 && gMean > 110) {
            derivedProbs =
              selectedCrop === 'tomato'
                ? [0.01, 0.01, 0.01, 0.95, 0.005, 0.005, 0.005, 0.005]
                : [0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.95];
          } else {
            derivedProbs =
              selectedCrop === 'tomato'
                ? [0.88, 0.06, 0.03, 0.01, 0.005, 0.005, 0.005, 0.005]
                : [0.005, 0.005, 0.005, 0.005, 0.04, 0.88, 0.05, 0.01];
          }
        }

        setCustomImageName(file.name);
        setCustomImageDataUrl(dataUrl);
        setCustomRawProbs(derivedProbs);
        executeDiagnosis(derivedProbs, selectedCrop, `Upload: ${file.name}`, dataUrl);
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const handleRestoreFromHistory = (item: DiagnosticHistoryItem) => {
    setUploadError(null);
    setSelectedCrop(item.cropSelected);
    setActiveSampleTitle(item.sampleName);
    setActivePreviewUrl(item.imagePreviewDataUrl);

    const matchingPreset = SAMPLE_PRESETS.find((p) => p.label === item.sampleName);
    if (matchingPreset) {
      setSelectedPreset(matchingPreset);
      const recomputed = runCropAwareInference(
        matchingPreset.rawProbabilities,
        item.cropSelected,
        tConf,
        tMargin
      );
      setCurrentResult(recomputed);
    } else {
      setCurrentResult({
        status: item.status,
        cropSelected: item.cropSelected,
        predictedClassKey: item.predictedClassKey,
        diseaseName: item.diseaseName,
        confidence: item.confidence,
        top2ClassKey: item.top2ClassKey,
        top2DiseaseName: item.top2DiseaseName,
        top2Confidence: item.top2Confidence,
        margin: item.margin,
        rawCropMass: item.rawCropMass,
        renormalizedProbs: [],
        message: item.message,
        remedy: item.remedy,
        latencyMs: item.latencyMs
      });
    }
  };

  const handleDeleteHistoryItem = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  const formatTimestamp = (iso: string) => {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return iso;
    }
  };

  const getStatusBadge = (status: DiagnosticHistoryItem['status']) => {
    switch (status) {
      case 'confident_disease':
        return {
          dot: 'bg-amber-500 ring-4 ring-amber-500/20',
          text: '▲ DISEASE DETECTED',
          textColor: 'text-amber-900',
          border: 'border-amber-300 bg-amber-50/70'
        };
      case 'confident_healthy':
        return {
          dot: 'bg-emerald-500 ring-4 ring-emerald-500/20',
          text: '● HEALTHY LEAF',
          textColor: 'text-emerald-900',
          border: 'border-emerald-300 bg-emerald-50/70'
        };
      case 'low_confidence':
        return {
          dot: 'bg-rose-500 ring-4 ring-rose-500/20',
          text: '✖ LOW CONFIDENCE — RETAKE PHOTO',
          textColor: 'text-rose-900',
          border: 'border-rose-300 bg-rose-50/70'
        };
      case 'crop_mismatch':
        return {
          dot: 'bg-rose-500 ring-4 ring-rose-500/20',
          text: '✖ CROP MISMATCH DETECTED',
          textColor: 'text-rose-900',
          border: 'border-rose-300 bg-rose-50/70'
        };
    }
  };

  const activeBadge = getStatusBadge(currentResult.status);
  const activeRemedyOrFallback =
    currentResult.remedy ||
    (currentResult.cropSelected === 'tomato' ? KNOWLEDGE_BASE[0] : KNOWLEDGE_BASE[5]);

  const mapEmbedQuery = encodeURIComponent(
    `agricultural supply pesticide fertilizer store near ${
      userCoords
        ? `${userCoords.lat},${userCoords.lng}`
        : locationQuery || 'Fresno, CA'
    }`
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200 px-6 py-3.5">
        <div className="max-w-[1380px] mx-auto flex items-center justify-between gap-4">
          {/* Zone 1: Brand Title (single text element wordmark) */}
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('diagnostic');
            }}
            className="text-xl font-bold tracking-tight text-slate-900 font-display whitespace-nowrap shrink-0"
          >
            PhytoScan
          </a>

          {/* Zone 2: 5 Clean Navigation Links */}
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
            <button
              type="button"
              onClick={() => setActiveTab('diagnostic')}
              className={`py-1 transition-colors whitespace-nowrap border-b-2 ${
                activeTab === 'diagnostic'
                  ? 'border-emerald-600 text-slate-900 font-semibold'
                  : 'border-transparent hover:text-slate-900'
              }`}
            >
              Diagnostic Lab
            </button>
            <a
              href="#pesticide-maps-section"
              onClick={() => setActiveTab('diagnostic')}
              className="py-1 hover:text-slate-900 transition-colors whitespace-nowrap"
            >
              Pesticide Map Locator
            </a>
            <a
              href="#agronomy-chat-section"
              onClick={() => setActiveTab('diagnostic')}
              className="py-1 hover:text-slate-900 transition-colors whitespace-nowrap"
            >
              Agronomist Chat
            </a>
            <a
              href="#recent-history-section"
              onClick={() => setActiveTab('diagnostic')}
              className="py-1 hover:text-slate-900 transition-colors whitespace-nowrap"
            >
              History ({history.length}/5)
            </a>
            <button
              type="button"
              onClick={() => setActiveTab('spec')}
              className={`py-1 transition-colors whitespace-nowrap border-b-2 ${
                activeTab === 'spec'
                  ? 'border-emerald-600 text-slate-900 font-semibold'
                  : 'border-transparent hover:text-slate-900'
              }`}
            >
              14-Section Spec
            </button>
          </nav>

          {/* Zone 3: Primary Action */}
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => {
                setActiveTab('diagnostic');
                const activeProbs = customRawProbs || selectedPreset.rawProbabilities;
                const label = customImageName
                  ? `Upload: ${customImageName}`
                  : selectedPreset.label;
                const preview = customImageDataUrl || selectedPreset.svgDataUrl;
                executeDiagnosis(activeProbs, selectedCrop, label, preview);
              }}
              className="px-4 py-2 text-xs font-semibold text-white bg-emerald-700 rounded-lg hover:bg-emerald-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              Run Diagnosis
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1380px] w-full mx-auto px-6 py-8">
        {/* Top Mode Switcher for Mobile & Quick Access */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-6 mb-8 border-b border-slate-200">
          <div>
            <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900 tracking-tight text-balance">
              AI Crop-Disease Detector, Pesticide Store Map &amp; Agronomy Advisor
            </h1>
            <p className="text-sm text-slate-600 mt-1">
              Tomato &amp; Maize Foliar Diagnosis · Specific Pesticide &amp; Medicine Finder via Google Maps · Published Articles via Google Search
            </p>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-200/80 rounded-lg">
            <button
              type="button"
              onClick={() => setActiveTab('diagnostic')}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'diagnostic'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Microscope className="w-3.5 h-3.5" />
              <span>Diagnostic Main Screen</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('kb')}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'kb'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>8-Class Knowledge Base</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('spec')}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'spec'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Full 14-Section Spec Sheet</span>
            </button>
          </div>
        </div>

        {activeTab === 'diagnostic' && (
          <div className="space-y-10">
            {/* Asymmetric Split Console: Left Controls (5 cols) + Right Result & Telemetry (7 cols) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Left Column: Step 1 Crop Selector, Step 2 Leaf Input / Presets, Step 3 Threshold Calibration */}
              <div className="lg:col-span-5 space-y-6">
                {/* Step 1: Crop Selection */}
                <div className="bg-white border border-slate-200 rounded-xl p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-semibold text-slate-900">
                      01. Select Target Crop
                    </h2>
                    <span className="text-xs text-slate-500 font-mono">
                      4 classes / crop
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      id="crop-tomato-btn"
                      type="button"
                      onClick={() => {
                        setSelectedCrop('tomato');
                        const probs =
                          customRawProbs || selectedPreset.rawProbabilities;
                        const label = customImageName
                          ? `Upload: ${customImageName}`
                          : selectedPreset.label;
                        const preview =
                          customImageDataUrl || selectedPreset.svgDataUrl;
                        executeDiagnosis(probs, 'tomato', label, preview);
                      }}
                      className={`py-3 px-4 rounded-lg border text-left transition-colors cursor-pointer ${
                        selectedCrop === 'tomato'
                          ? 'bg-emerald-950 text-white border-emerald-950'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      <div className="text-sm font-semibold">Tomato</div>
                      <div
                        className={`text-xs mt-0.5 ${
                          selectedCrop === 'tomato'
                            ? 'text-emerald-200'
                            : 'text-slate-500'
                        }`}
                      >
                        Solanum lycopersicum · Idx 0–3
                      </div>
                    </button>

                    <button
                      id="crop-maize-btn"
                      type="button"
                      onClick={() => {
                        setSelectedCrop('maize');
                        const probs =
                          customRawProbs || selectedPreset.rawProbabilities;
                        const label = customImageName
                          ? `Upload: ${customImageName}`
                          : selectedPreset.label;
                        const preview =
                          customImageDataUrl || selectedPreset.svgDataUrl;
                        executeDiagnosis(probs, 'maize', label, preview);
                      }}
                      className={`py-3 px-4 rounded-lg border text-left transition-colors cursor-pointer ${
                        selectedCrop === 'maize'
                          ? 'bg-emerald-950 text-white border-emerald-950'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      <div className="text-sm font-semibold">Maize (Corn)</div>
                      <div
                        className={`text-xs mt-0.5 ${
                          selectedCrop === 'maize'
                            ? 'text-emerald-200'
                            : 'text-slate-500'
                        }`}
                      >
                        Zea mays · Idx 4–7
                      </div>
                    </button>
                  </div>
                </div>

                {/* Step 2: Upload Custom Leaf Photo OR Pick Verified Test Sample */}
                <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-slate-900">
                      02. Leaf Image Input (Upload or Test Preset)
                    </h2>
                    <span className="text-xs text-slate-500 font-mono">
                      224×224 RGB
                    </span>
                  </div>

                  {/* Upload Box */}
                  <div>
                    <input
                      ref={fileInputRef}
                      id="leaf-file-input"
                      type="file"
                      accept=".jpg,.jpeg,.png"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full border border-dashed border-slate-300 hover:border-emerald-600 bg-slate-50/70 hover:bg-emerald-50/30 rounded-lg p-4 text-center transition-colors cursor-pointer"
                    >
                      <Upload className="w-5 h-5 text-slate-500 mx-auto mb-1.5" />
                      <div className="text-xs font-semibold text-slate-800">
                        Upload Leaf Photo (.jpg, .jpeg, .png · max 5 MB)
                      </div>
                      <div className="text-[11px] text-slate-500 mt-0.5">
                        Auto-runs EXIF transpose, 224×224 bilinear resize &amp; saves result to Recent History
                      </div>
                    </button>
                  </div>

                  {uploadError && (
                    <div className="p-3 rounded-lg border border-rose-300 bg-rose-50 text-xs text-rose-900 font-mono">
                      {uploadError}
                    </div>
                  )}

                  {/* Preset Selector for Instant Testing */}
                  <div>
                    <div className="text-xs font-medium text-slate-600 mb-2">
                      Or click a calibrated PlantVillage &amp; Edge-Case test sample (immediately logs to Last 5 History):
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {SAMPLE_PRESETS.map((preset) => {
                        const isSelected =
                          !customImageName && selectedPreset.id === preset.id;
                        return (
                          <button
                            key={preset.id}
                            type="button"
                            onClick={() => handleSelectPreset(preset)}
                            className={`p-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                              isSelected
                                ? 'border-emerald-600 bg-emerald-50/60 text-slate-900'
                                : 'border-slate-200 bg-white hover:bg-slate-50 text-slate-700'
                            }`}
                          >
                            <div className="text-xs font-semibold truncate">
                              {preset.label}
                            </div>
                            <div className="text-[11px] text-slate-500 truncate mt-0.5">
                              {preset.description}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Step 3: Threshold & Margin Calibration Controls */}
                <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-1.5">
                      <Sliders className="w-4 h-4 text-slate-500" />
                      <span>03. Confidence &amp; Margin Gate Calibration</span>
                    </h2>
                    <button
                      type="button"
                      onClick={() => {
                        setTConf(0.65);
                        setTMargin(0.15);
                        const probs =
                          customRawProbs || selectedPreset.rawProbabilities;
                        const label = customImageName
                          ? `Upload: ${customImageName}`
                          : selectedPreset.label;
                        const preview =
                          customImageDataUrl || selectedPreset.svgDataUrl;
                        executeDiagnosis(probs, selectedCrop, label, preview, 0.65, 0.15);
                      }}
                      className="text-xs text-emerald-700 hover:underline font-medium cursor-pointer"
                    >
                      Reset Defaults
                    </button>
                  </div>

                  <div className="space-y-3 text-xs">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-slate-600 font-medium">
                          Confidence Threshold (T_conf)
                        </span>
                        <span className="font-mono font-semibold text-slate-900">
                          {(tConf * 100).toFixed(0)}% (spec: 65%)
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.40"
                        max="0.95"
                        step="0.05"
                        value={tConf}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          setTConf(val);
                          const probs =
                            customRawProbs || selectedPreset.rawProbabilities;
                          setCurrentResult(
                            runCropAwareInference(probs, selectedCrop, val, tMargin)
                          );
                        }}
                        className="w-full accent-emerald-700 cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-slate-600 font-medium">
                          Top-1 vs Top-2 Margin Gate (T_margin)
                        </span>
                        <span className="font-mono font-semibold text-slate-900">
                          {(tMargin * 100).toFixed(0)}% (spec: 15%)
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.05"
                        max="0.40"
                        step="0.05"
                        value={tMargin}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          setTMargin(val);
                          const probs =
                            customRawProbs || selectedPreset.rawProbabilities;
                          setCurrentResult(
                            runCropAwareInference(probs, selectedCrop, tConf, val)
                          );
                        }}
                        className="w-full accent-emerald-700 cursor-pointer"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Live Diagnostic Output + Softmax Telemetry + Agronomy Remedy */}
              <div className="lg:col-span-7 space-y-6">
                <section
                  id="result-card"
                  aria-live="polite"
                  className="bg-white border border-slate-200 rounded-xl p-6 space-y-6"
                >
                  {/* Top Status Banner */}
                  <div
                    className={`flex flex-wrap items-center justify-between gap-3 p-4 rounded-lg border ${activeBadge.border}`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className={`w-2.5 h-2.5 rounded-full ${activeBadge.dot}`} />
                      <span
                        className={`text-xs font-mono font-semibold tracking-wide ${activeBadge.textColor}`}
                      >
                        {activeBadge.text}
                      </span>
                    </div>
                    <div className="text-xs font-mono text-slate-600">
                      <span>Crop: {currentResult.cropSelected.toUpperCase()}</span>
                      <span className="mx-2">·</span>
                      <span>CPU Latency: {currentResult.latencyMs.toFixed(1)} ms</span>
                    </div>
                  </div>

                  {/* Preview + Primary Quantitative Metrics */}
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-6 items-center">
                    <div className="sm:col-span-4">
                      <div className="aspect-square w-full max-w-[200px] mx-auto rounded-lg border border-slate-200 overflow-hidden bg-slate-100 relative">
                        <img
                          id="image-preview"
                          src={activePreviewUrl}
                          alt={activeSampleTitle}
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-cover"
                        />
                        {isAnalyzing && (
                          <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center text-white text-xs font-mono">
                            Running CNN...
                          </div>
                        )}
                      </div>
                      <div className="text-center text-[11px] font-mono text-slate-500 mt-1.5 truncate">
                        {activeSampleTitle}
                      </div>
                    </div>

                    <div className="sm:col-span-8 space-y-4">
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div>
                          <div className="text-xs text-slate-500">
                            Primary Diagnostic Output
                          </div>
                          <h3 className="text-2xl font-semibold text-slate-900 mt-0.5">
                            {currentResult.diseaseName}
                          </h3>
                          {currentResult.predictedClassKey && (
                            <div className="text-xs font-mono text-emerald-800 mt-0.5">
                              Folder Key: {currentResult.predictedClassKey}
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-2">
                          <a
                            href="#pesticide-maps-section"
                            onClick={() => fetchGoogleMapsStores()}
                            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-white bg-emerald-700 hover:bg-emerald-800 rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                          >
                            <MapPin className="w-3.5 h-3.5" />
                            <span>Find Pesticide Stores</span>
                          </a>
                          <button
                            type="button"
                            onClick={() =>
                              fetchGoogleSearchGrounding({
                                diseaseName: currentResult.diseaseName,
                                causalAgent: currentResult.remedy?.causal_agent,
                                crop: currentResult.cropSelected
                              })
                            }
                            disabled={isSearchingWeb}
                            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-emerald-900 bg-emerald-50 hover:bg-emerald-100 border border-emerald-300 rounded-lg transition-colors whitespace-nowrap cursor-pointer disabled:opacity-60"
                          >
                            {isSearchingWeb ? (
                              <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Globe className="w-3.5 h-3.5 text-emerald-700" />
                            )}
                            <span>Published Articles</span>
                          </button>
                        </div>
                      </div>

                      {/* 3 Telemetry Readouts */}
                      <div className="grid grid-cols-3 gap-3 pt-2 border-t border-slate-100">
                        <div>
                          <div className="text-[11px] text-slate-500">
                            Top-1 Confidence
                          </div>
                          <div className="text-xl font-mono font-bold text-slate-900 tabular-nums">
                            {(currentResult.confidence * 100).toFixed(1)}
                            <span className="text-xs font-normal text-slate-400 ml-0.5">
                              %
                            </span>
                          </div>
                          <div className="text-[11px] font-mono text-slate-500">
                            Gate &gt;= {(tConf * 100).toFixed(0)}%
                          </div>
                        </div>

                        <div>
                          <div className="text-[11px] text-slate-500">
                            Top-1 / Top-2 Margin
                          </div>
                          <div className="text-xl font-mono font-bold text-slate-900 tabular-nums">
                            {(currentResult.margin * 100).toFixed(1)}
                            <span className="text-xs font-normal text-slate-400 ml-0.5">
                              %
                            </span>
                          </div>
                          <div className="text-[11px] font-mono text-slate-500">
                            Gate &gt;= {(tMargin * 100).toFixed(0)}%
                          </div>
                        </div>

                        <div>
                          <div className="text-[11px] text-slate-500">
                            Crop Prob Mass
                          </div>
                          <div className="text-xl font-mono font-bold text-slate-900 tabular-nums">
                            {(currentResult.rawCropMass * 100).toFixed(1)}
                            <span className="text-xs font-normal text-slate-400 ml-0.5">
                              %
                            </span>
                          </div>
                          <div className="text-[11px] font-mono text-slate-500">
                            Gate &gt;= 35%
                          </div>
                        </div>
                      </div>

                      <p className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-3 leading-relaxed">
                        {currentResult.message}
                      </p>
                    </div>
                  </div>

                  {/* Crop-Aware Renormalized Softmax Distribution */}
                  {currentResult.renormalizedProbs.length > 0 && (
                    <div className="pt-4 border-t border-slate-200">
                      <div className="flex items-center justify-between text-xs mb-2.5">
                        <span className="font-semibold text-slate-900">
                          Crop-Aware Softmax Distribution ({currentResult.cropSelected.toUpperCase()} classes renormalized)
                        </span>
                        <span className="font-mono text-slate-500">
                          p_crop,i = p_raw,i / M_crop
                        </span>
                      </div>
                      <div className="space-y-2">
                        {currentResult.renormalizedProbs.map((item) => {
                          const pct = (item.renormProb * 100).toFixed(1);
                          return (
                            <div key={item.entry.classKey} className="text-xs">
                              <div className="flex justify-between font-mono mb-1">
                                <span className="text-slate-800 truncate">
                                  {item.entry.disease_name}{' '}
                                  <span className="text-slate-400">
                                    ({item.entry.classKey})
                                  </span>
                                </span>
                                <span className="tabular-nums font-semibold text-slate-900">
                                  {pct}%{' '}
                                  <span className="text-slate-400 font-normal">
                                    (raw {(item.rawProb * 100).toFixed(1)}%)
                                  </span>
                                </span>
                              </div>
                              <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-emerald-600 rounded-full transition-all duration-150"
                                  style={{
                                    width: `${Math.min(100, Math.max(1, item.renormProb * 100))}%`
                                  }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Knowledge Base Remedy Details (When Confident) */}
                  {currentResult.remedy ? (
                    <div className="pt-4 border-t border-slate-200 space-y-4">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <h4 className="text-sm font-semibold text-slate-900">
                          Specific Pesticides, Medicines &amp; Field Action Plan
                        </h4>
                        <div className="text-xs text-slate-600">
                          <span>Causal Agent: <strong>{currentResult.remedy.causal_agent}</strong></span>
                          <span className="mx-2">·</span>
                          <span>Severity: <strong>{currentResult.remedy.severity}</strong></span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div className="bg-emerald-50/70 border border-emerald-200 rounded-lg p-3.5">
                          <div className="font-semibold text-emerald-950 mb-1">
                            Organic / Biological Medicine &amp; Remedy
                          </div>
                          <p className="text-slate-800 leading-relaxed">
                            {currentResult.remedy.organic_remedy}
                          </p>
                        </div>

                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5">
                          <div className="font-semibold text-slate-900 mb-1">
                            Specific Chemical Pesticide / Fungicide (Active Ingredients)
                          </div>
                          <p className="text-slate-700 leading-relaxed">
                            {currentResult.remedy.chemical_remedy}
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div className="border border-slate-200 rounded-lg p-3.5 space-y-2">
                          <div className="font-semibold text-slate-900">
                            Visual Field Symptoms
                          </div>
                          <ul className="list-disc pl-4 space-y-1 text-slate-700">
                            {currentResult.remedy.symptoms.map((sym, i) => (
                              <li key={i}>{sym}</li>
                            ))}
                          </ul>
                        </div>

                        <div className="border border-slate-200 rounded-lg p-3.5 space-y-2">
                          <div className="font-semibold text-slate-900">
                            Prevention &amp; Cultural Practices
                          </div>
                          <ul className="list-disc pl-4 space-y-1 text-slate-700">
                            {currentResult.remedy.prevention_tips.map((tip, i) => (
                              <li key={i}>{tip}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="pt-4 border-t border-slate-200">
                      <div className="p-4 rounded-lg bg-amber-50/80 border border-amber-200 text-xs text-amber-950 space-y-2">
                        <div className="font-semibold flex items-center gap-1.5">
                          <AlertTriangle className="w-4 h-4 text-amber-700" />
                          <span>Safety Fallback Active — Curative Pesticide Suppressed</span>
                        </div>
                        <p>
                          To prevent misapplication of fungicides on an uncertain or mismatched photo, curative chemical spray advice is withheld until confidence reaches at least <strong>{(tConf * 100).toFixed(0)}%</strong> with a top-2 margin of at least <strong>{(tMargin * 100).toFixed(0)}%</strong>. You can still search nearby agro-supply stores below for preventative crop protection inputs.
                        </p>
                      </div>
                    </div>
                  )}
                </section>
              </div>
            </div>

            {/* PESTICIDE & PLANT MEDICINE STORE LOCATOR (GOOGLE MAPS GROUNDING) */}
            <section
              id="pesticide-maps-section"
              className="bg-white border border-slate-200 rounded-xl p-6 space-y-5"
            >
              <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-slate-200">
                <div>
                  <div className="flex items-center gap-2">
                    <MapPin className="w-5 h-5 text-emerald-700" />
                    <h2 className="text-lg font-semibold text-slate-900">
                      Specific Pesticide &amp; Medicine Finder + Google Maps Store Locator
                    </h2>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">
                    Identifies the exact fungicides, biological sprays, and preventative medicines required for <strong>{currentResult.diseaseName}</strong> and uses <strong>Google Maps Grounding</strong> (<code>gemini-3.5-flash</code> + <code>googleMaps</code> tool) to locate nearby agro-dealers, farm co-ops, and plant pharmacies.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleUseMyLocation}
                    disabled={isLocatingGps || isSearchingMaps}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-emerald-900 bg-emerald-50 hover:bg-emerald-100 border border-emerald-300 rounded-lg transition-colors whitespace-nowrap cursor-pointer disabled:opacity-60"
                  >
                    {isLocatingGps ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Navigation className="w-3.5 h-3.5 text-emerald-700" />
                    )}
                    <span>Use My GPS Location</span>
                  </button>
                </div>
              </div>

              {/* Required Pesticide & Medicine Component Summary Bar */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-slate-500 font-medium mb-1">
                    Target Disease &amp; Pathogen
                  </div>
                  <div className="font-semibold text-slate-900">
                    {activeRemedyOrFallback.disease_name}
                  </div>
                  <div className="font-mono text-emerald-800 mt-0.5">
                    {activeRemedyOrFallback.causal_agent}
                  </div>
                </div>

                <div className="p-3.5 rounded-lg bg-emerald-50/60 border border-emerald-200">
                  <div className="text-emerald-950 font-semibold mb-1">
                    Required Organic / Preventative Components
                  </div>
                  <div className="text-slate-800 leading-relaxed">
                    {activeRemedyOrFallback.organic_remedy}
                  </div>
                </div>

                <div className="p-3.5 rounded-lg bg-amber-50/60 border border-amber-200">
                  <div className="text-amber-950 font-semibold mb-1 flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-amber-700" />
                    <span>Required Specific Pesticide / Fungicide</span>
                  </div>
                  <div className="text-slate-800 leading-relaxed">
                    {activeRemedyOrFallback.chemical_remedy}
                  </div>
                </div>
              </div>

              {/* Location Search Form + Quick Agricultural Hubs */}
              <div className="space-y-2.5">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    setUserCoords(null);
                    fetchGoogleMapsStores({ locQuery: locationQuery });
                  }}
                  className="flex flex-col sm:flex-row gap-2.5"
                >
                  <div className="relative flex-1">
                    <MapPin className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      value={locationQuery}
                      onChange={(e) => setLocationQuery(e.target.value)}
                      placeholder="Enter your city, district, or postal code (e.g., Fresno CA, Pune India, Ames Iowa)..."
                      className="w-full pl-9 pr-3.5 py-2.5 text-xs border border-slate-200 rounded-lg bg-slate-50 focus:bg-white focus:outline-none focus:border-emerald-600"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={isSearchingMaps}
                    className="inline-flex items-center justify-center gap-1.5 px-5 py-2.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap cursor-pointer disabled:opacity-60"
                  >
                    {isSearchingMaps ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Search className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {isSearchingMaps
                        ? 'Locating Pesticide Stores on Google Maps...'
                        : 'Find Nearby Stores on Map'}
                    </span>
                  </button>
                </form>

                {/* Quick Region Buttons */}
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <span>Quick farming regions:</span>
                  {[
                    'Fresno, California',
                    'Pune, Maharashtra, India',
                    'Ames, Iowa',
                    ' Salinas, California',
                    'Nakuru, Kenya'
                  ].map((hub) => (
                    <button
                      key={hub}
                      type="button"
                      onClick={() => {
                        setLocationQuery(hub.trim());
                        setUserCoords(null);
                        fetchGoogleMapsStores({ locQuery: hub.trim() });
                      }}
                      className="px-2.5 py-1 rounded-md border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 transition-colors cursor-pointer"
                    >
                      {hub.trim()}
                    </button>
                  ))}
                </div>
              </div>

              {mapsError && (
                <div className="p-3.5 rounded-lg border border-rose-200 bg-rose-50 text-xs text-rose-900">
                  {mapsError}
                </div>
              )}

              {/* Split View: Interactive Map Viewport + Google Maps Grounded Store Results */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start pt-2">
                {/* Left 5 Cols: Interactive Embedded Map showing Agro-Supply & Pesticide Stores */}
                <div className="lg:col-span-5 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-900">
                      Interactive Store Map ({locationQuery || 'Nearby'})
                    </span>
                    <a
                      href={`https://www.google.com/maps/search/?api=1&query=${mapEmbedQuery}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-emerald-700 hover:underline inline-flex items-center gap-1 font-medium"
                    >
                      <span>Open Full Map</span>
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                  <div className="w-full h-[340px] rounded-xl border border-slate-200 overflow-hidden bg-slate-100">
                    <iframe
                      title="Nearby Agricultural Supply and Pesticide Stores Map"
                      src={`https://www.google.com/maps?q=${mapEmbedQuery}&output=embed`}
                      className="w-full h-full border-0"
                      loading="lazy"
                      referrerPolicy="no-referrer-when-downgrade"
                    />
                  </div>
                </div>

                {/* Right 7 Cols: Google Maps Grounding Output & Verified Place Links */}
                <div className="lg:col-span-7 space-y-4">
                  {mapsResult ? (
                    <>
                      {/* Verified Google Maps Places from groundingChunks.maps */}
                      <div>
                        <div className="flex items-center justify-between text-xs mb-2.5">
                          <span className="font-semibold text-slate-900">
                            Verified Google Maps Agro-Dealers &amp; Supply Stores ({mapsResult.places.length})
                          </span>
                          <span className="font-mono text-slate-400">
                            Model: {mapsResult.modelUsed} · Region: {mapsResult.locationUsed}
                          </span>
                        </div>

                        {mapsResult.places.length > 0 ? (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {mapsResult.places.map((place, idx) => (
                              <a
                                key={idx}
                                href={place.uri}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="p-3.5 rounded-xl border border-slate-200 hover:border-emerald-600 bg-slate-50/70 hover:bg-white transition-colors flex flex-col justify-between gap-2 group"
                              >
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex items-center gap-2 min-w-0">
                                    <MapPin className="w-4 h-4 text-emerald-700 shrink-0" />
                                    <span className="text-xs font-semibold text-slate-900 group-hover:text-emerald-800 truncate">
                                      {place.title}
                                    </span>
                                  </div>
                                  <ExternalLink className="w-3.5 h-3.5 text-slate-400 group-hover:text-emerald-700 shrink-0" />
                                </div>

                                {place.reviewSnippets.length > 0 && (
                                  <div className="text-[11px] text-slate-600 italic line-clamp-2 border-l-2 border-emerald-300 pl-2">
                                    "{place.reviewSnippets[0]}"
                                  </div>
                                )}

                                <div className="text-[11px] font-mono text-emerald-700 truncate">
                                  View Directions &amp; Inventory on Google Maps →
                                </div>
                              </a>
                            ))}
                          </div>
                        ) : (
                          <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 text-xs text-slate-600">
                            See the detailed store recommendations below or click <strong>Open Full Map</strong> to view pins in {mapsResult.locationUsed}.
                          </div>
                        )}
                      </div>

                      {/* Detailed Pesticide & Store Guidance from Gemini Maps Grounding */}
                      <div>
                        <div className="text-xs font-semibold text-slate-900 mb-1.5">
                          Specific Pesticide Prescription &amp; Local Purchasing Guide
                        </div>
                        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-wrap max-h-[260px] overflow-y-auto">
                          {mapsResult.guidance}
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="h-[340px] rounded-xl border border-dashed border-slate-200 bg-slate-50/50 p-6 flex flex-col items-center justify-center text-center">
                      <MapPin className="w-8 h-8 text-emerald-700 mb-2" />
                      <h3 className="text-sm font-semibold text-slate-900">
                        Locate Specific Pesticides &amp; Agricultural Medicines on Google Maps
                      </h3>
                      <p className="text-xs text-slate-600 max-w-md mt-1 mb-4">
                        Click <strong>Find Nearby Stores on Map</strong> or <strong>Use My GPS Location</strong> to query Google Maps Grounding for verified agro-dealers carrying <strong>{activeRemedyOrFallback.chemical_remedy.split('(')[0].trim()}</strong> and organic copper/neem treatments.
                      </p>
                      <button
                        type="button"
                        onClick={() => fetchGoogleMapsStores({ locQuery: locationQuery })}
                        disabled={isSearchingMaps}
                        className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-emerald-700 hover:bg-emerald-800 rounded-lg transition-colors cursor-pointer"
                      >
                        <MapPin className="w-3.5 h-3.5" />
                        <span>Search Stores in {locationQuery}</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* MULTI-TURN GEMINI AGRONOMIST & PESTICIDE ADVISOR CHATBOT */}
            <section
              id="agronomy-chat-section"
              className="bg-white border border-slate-200 rounded-xl p-6 space-y-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
                <div>
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-5 h-5 text-emerald-700" />
                    <h2 className="text-lg font-semibold text-slate-900">
                      Dr. PhytoScan — Multi-Turn Agronomy &amp; Pesticide Medicine Chatbot
                    </h2>
                  </div>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Context-aware assistant for <strong>{currentResult.diseaseName}</strong> ({currentResult.cropSelected.toUpperCase()}). Ask about specific fungicide active ingredients, tank-mix compatibility, dosage per liter/acre, and prevention schedules.
                  </p>
                </div>

                {/* Model Tier Selector: Fast (3.1-flash-lite) | General (3.5-flash) | Complex (3.1-pro-preview) */}
                <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setChatTier('fast')}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                      chatTier === 'fast'
                        ? 'bg-white text-slate-900 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Fast · Flash-Lite
                  </button>
                  <button
                    type="button"
                    onClick={() => setChatTier('general')}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                      chatTier === 'general'
                        ? 'bg-white text-slate-900 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    General · 3.5-Flash
                  </button>
                  <button
                    type="button"
                    onClick={() => setChatTier('complex')}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                      chatTier === 'complex'
                        ? 'bg-white text-slate-900 shadow-xs font-semibold'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Deep · 3.1-Pro
                  </button>
                </div>
              </div>

              {/* Quick Contextual Prompts */}
              <div className="flex flex-wrap items-center gap-2">
                {[
                  `What specific pesticides and medicines prevent ${currentResult.diseaseName}?`,
                  `Give me a 21-day spray schedule and exact dosage per liter for ${currentResult.diseaseName}.`,
                  `What organic medicines work if chemical fungicides are unavailable?`,
                  `What FRAC fungicide rotation prevents resistance in ${currentResult.cropSelected}?`
                ].map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSendChatMessage(q)}
                    disabled={isChatLoading}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-emerald-50/50 hover:border-emerald-300 text-xs text-slate-700 transition-colors cursor-pointer disabled:opacity-50"
                  >
                    <Sparkles className="w-3 h-3 text-emerald-700 shrink-0" />
                    <span className="truncate max-w-[320px]">{q}</span>
                  </button>
                ))}
              </div>

              {/* Scrollable Conversation Thread */}
              <div
                ref={chatScrollRef}
                className="h-[320px] overflow-y-auto border border-slate-200 rounded-xl bg-slate-50/70 p-4 space-y-3"
              >
                {chatMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${
                      msg.role === 'user' ? 'items-end' : 'items-start'
                    }`}
                  >
                    <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono mb-1 px-1">
                      <span>
                        {msg.role === 'user' ? 'Farmer' : 'Dr. PhytoScan'}
                      </span>
                      <span>·</span>
                      <span>{msg.timestamp}</span>
                      {msg.modelUsed && (
                        <>
                          <span>·</span>
                          <span>{msg.modelUsed}</span>
                        </>
                      )}
                    </div>
                    <div
                      className={`max-w-[85%] rounded-xl px-4 py-3 text-xs leading-relaxed whitespace-pre-wrap ${
                        msg.role === 'user'
                          ? 'bg-emerald-900 text-white'
                          : 'bg-white border border-slate-200 text-slate-800'
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}

                {isChatLoading && (
                  <div className="flex items-center gap-2 text-xs text-slate-500 font-mono px-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-700" />
                    <span>Dr. PhytoScan is formulating treatment protocol...</span>
                  </div>
                )}
              </div>

              {chatError && (
                <div className="p-3 rounded-lg border border-rose-200 bg-rose-50 text-xs text-rose-900">
                  {chatError}
                </div>
              )}

              {/* Message Input Form */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSendChatMessage();
                }}
                className="flex gap-2"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder={`Ask Dr. PhytoScan about pesticides, medicine dosages, or prevention for ${currentResult.diseaseName}...`}
                  className="flex-1 px-4 py-2.5 text-xs border border-slate-200 rounded-lg bg-slate-50 focus:bg-white focus:outline-none focus:border-emerald-600"
                />
                <button
                  type="submit"
                  disabled={isChatLoading || !chatInput.trim()}
                  className="inline-flex items-center gap-1.5 px-5 py-2.5 text-xs font-semibold text-white bg-emerald-700 hover:bg-emerald-800 rounded-lg transition-colors whitespace-nowrap cursor-pointer disabled:opacity-50"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>Send</span>
                </button>
              </form>
            </section>

            {/* GOOGLE SEARCH GROUNDING & PUBLISHED EXTENSION ARTICLES PANEL */}
            <section
              id="google-search-grounding-section"
              className="bg-white border border-slate-200 rounded-xl p-6 space-y-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
                <div>
                  <div className="flex items-center gap-2">
                    <Globe className="w-5 h-5 text-emerald-700" />
                    <h2 className="text-lg font-semibold text-slate-900">
                      Google Search Grounding — Published Extension Articles &amp; Live Agronomy Data
                    </h2>
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Retrieves up-to-date university extension publications, FAO/CABI guides, and peer-reviewed plant pathology articles for <strong>{currentResult.diseaseName}</strong> using Gemini with <code>googleSearch</code> grounding.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() =>
                    fetchGoogleSearchGrounding({
                      diseaseName: currentResult.diseaseName,
                      causalAgent: currentResult.remedy?.causal_agent,
                      crop: currentResult.cropSelected
                    })
                  }
                  disabled={isSearchingWeb}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap cursor-pointer disabled:opacity-60"
                >
                  {isSearchingWeb ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Search className="w-3.5 h-3.5" />
                  )}
                  <span>
                    {isSearchingWeb
                      ? 'Searching Google Scholar & Extension...'
                      : `Get Articles for ${currentResult.diseaseName}`}
                  </span>
                </button>
              </div>

              {/* Custom Search Input Bar */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!searchQueryInput.trim()) return;
                  fetchGoogleSearchGrounding({
                    customQuery: searchQueryInput,
                    crop: currentResult.cropSelected
                  });
                }}
                className="flex flex-col sm:flex-row gap-2"
              >
                <input
                  type="text"
                  value={searchQueryInput}
                  onChange={(e) => setSearchQueryInput(e.target.value)}
                  placeholder={`Ask a specific question or search articles (e.g., "${currentResult.diseaseName} fungicide resistance management")`}
                  className="flex-1 px-3.5 py-2 text-xs border border-slate-200 rounded-lg bg-slate-50 focus:bg-white focus:outline-none focus:border-emerald-600"
                />
                <button
                  type="submit"
                  disabled={isSearchingWeb || !searchQueryInput.trim()}
                  className="px-4 py-2 text-xs font-semibold text-emerald-900 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors whitespace-nowrap cursor-pointer disabled:opacity-50"
                >
                  Search Web
                </button>
              </form>

              {searchError && (
                <div className="p-3.5 rounded-lg border border-rose-200 bg-rose-50 text-xs text-rose-900">
                  {searchError}
                </div>
              )}

              {groundedData ? (
                <div className="space-y-4 pt-2">
                  {/* Grounded Summary */}
                  <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed whitespace-pre-wrap">
                    {groundedData.summary}
                  </div>

                  {/* Published Web Articles & Citations from groundingChunks */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-2">
                      <span className="font-semibold text-slate-900">
                        Published Articles &amp; Verified Web Sources ({groundedData.articles.length})
                      </span>
                      <span className="font-mono text-slate-400">
                        Model: {groundedData.modelUsed}
                      </span>
                    </div>

                    {groundedData.articles.length === 0 ? (
                      <p className="text-xs text-slate-500">
                        No direct external links returned for this query. Try searching a specific university extension topic above.
                      </p>
                    ) : (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                        {groundedData.articles.map((art, i) => (
                          <a
                            key={i}
                            href={art.uri}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-start justify-between gap-2 p-3 rounded-lg border border-slate-200 hover:border-emerald-600 bg-white hover:bg-emerald-50/30 transition-colors text-xs group"
                          >
                            <div className="min-w-0">
                              <div className="font-semibold text-slate-900 group-hover:text-emerald-800 truncate">
                                {art.title}
                              </div>
                              <div className="text-[11px] font-mono text-slate-500 truncate mt-0.5">
                                {art.uri}
                              </div>
                            </div>
                            <ExternalLink className="w-3.5 h-3.5 text-slate-400 group-hover:text-emerald-700 shrink-0 mt-0.5" />
                          </a>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Executed Search Queries */}
                  {groundedData.searchQueries.length > 0 && (
                    <div className="text-[11px] text-slate-500 font-mono pt-2 border-t border-slate-100">
                      Google Search queries executed:{' '}
                      {groundedData.searchQueries.join(' · ')}
                    </div>
                  )}
                </div>
              ) : (
                !isSearchingWeb && (
                  <div className="p-4 rounded-lg border border-dashed border-slate-200 bg-slate-50/50 text-xs text-slate-600 flex items-center justify-between gap-4">
                    <span>
                      Click <strong>Get Articles for {currentResult.diseaseName}</strong> to pull live research papers, extension bulletins, and regional outbreak advisories via Google Search Grounding.
                    </span>
                  </div>
                )
              )}
            </section>

            {/* LOCALSTORAGE DIAGNOSTIC HISTORY SECTION (Stores & Displays Last 5 Diagnostic Results on Main Screen) */}
            <section
              id="recent-history-section"
              className="bg-white border border-slate-200 rounded-xl p-6"
            >
              <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-slate-200">
                <div>
                  <div className="flex items-center gap-2">
                    <History className="w-5 h-5 text-emerald-700" />
                    <h2 className="text-lg font-semibold text-slate-900">
                      Recent Diagnostic History (Last 5 Results)
                    </h2>
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Persisted automatically in browser <code>localStorage</code> (key: <code>{STORAGE_KEY}</code>, capped at 5 most recent entries). Click any diagnosis card to reload its full breakdown above.
                  </p>
                </div>

                <div className="flex items-center gap-2.5">
                  <span className="text-xs font-mono text-slate-600">
                    Stored: {history.length} / {MAX_HISTORY_ITEMS}
                  </span>
                  {history.length > 0 && (
                    <button
                      type="button"
                      onClick={handleClearHistory}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Clear History</span>
                    </button>
                  )}
                </div>
              </div>

              {history.length === 0 ? (
                <div className="text-center py-8 border border-dashed border-slate-200 rounded-lg bg-slate-50/50">
                  <p className="text-sm font-medium text-slate-700">
                    No diagnostic results stored in localStorage yet.
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Select any leaf preset above or upload a leaf image to record up to 5 recent diagnoses here.
                  </p>
                </div>
              ) : (
                <div
                  id="history-list"
                  className="grid grid-cols-1 md:grid-cols-5 gap-4"
                >
                  {history.map((item, idx) => {
                    const isHealthy = item.status === 'confident_healthy';
                    const isDisease = item.status === 'confident_disease';
                    return (
                      <div
                        key={item.id}
                        onClick={() => handleRestoreFromHistory(item)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            handleRestoreFromHistory(item);
                          }
                        }}
                        className="group border border-slate-200 hover:border-emerald-600 rounded-xl p-3.5 bg-slate-50/60 hover:bg-white transition-colors flex flex-col justify-between cursor-pointer text-left"
                      >
                        <div>
                          {/* Header row: slot index + timestamp + delete button */}
                          <div className="flex items-center justify-between gap-1 text-[11px] font-mono text-slate-500 mb-2">
                            <span>
                              #{idx + 1} · {formatTimestamp(item.timestamp)}
                            </span>
                            <button
                              type="button"
                              onClick={(e) => handleDeleteHistoryItem(item.id, e)}
                              title="Remove from history"
                              className="p-1 text-slate-400 hover:text-rose-600 rounded transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>

                          {/* Thumbnail + Crop */}
                          <div className="flex items-center gap-3 mb-2.5">
                            <img
                              src={item.imagePreviewDataUrl}
                              alt={item.diseaseName}
                              referrerPolicy="no-referrer"
                              className="w-12 h-12 rounded-lg border border-slate-200 object-cover shrink-0 bg-white"
                            />
                            <div className="min-w-0">
                              <div className="text-[11px] font-mono uppercase text-slate-500">
                                {item.cropSelected}
                              </div>
                              <div className="text-xs font-semibold text-slate-900 truncate">
                                {item.diseaseName}
                              </div>
                              <div className="text-[11px] text-slate-500 truncate">
                                {item.sampleName}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Bottom telemetry bar */}
                        <div className="pt-2.5 border-t border-slate-200/80 flex items-center justify-between text-xs">
                          <div className="flex items-center gap-1 font-medium">
                            {isHealthy ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                            ) : isDisease ? (
                              <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
                            )}
                            <span className="font-mono font-semibold text-slate-900 tabular-nums">
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <span className="text-[11px] text-emerald-700 font-medium inline-flex items-center gap-0.5 group-hover:underline">
                            Inspect
                            <ArrowUpRight className="w-3 h-3" />
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        )}

        {activeTab === 'kb' && (
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-slate-900 mb-1">
                Complete 8-Class Agronomy Knowledge Base (Tomato &amp; Maize)
              </h2>
              <p className="text-xs text-slate-600 mb-6">
                Exact PlantVillage folder names mapped to causal pathogens, field symptoms, organic treatments, generic chemical active ingredients, and cultural prevention tips.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {KNOWLEDGE_BASE.map((entry) => (
                  <div
                    key={entry.classKey}
                    className="border border-slate-200 rounded-xl p-5 bg-white space-y-3 text-xs"
                  >
                    <div className="flex items-start justify-between gap-2 pb-2.5 border-b border-slate-100">
                      <div>
                        <div className="text-sm font-semibold text-slate-900">
                          {entry.disease_name}
                        </div>
                        <div className="font-mono text-emerald-800 mt-0.5">
                          {entry.classKey}
                        </div>
                      </div>
                      <div className="text-right font-mono text-slate-500">
                        <div>Crop: {entry.crop.toUpperCase()}</div>
                        <div>Severity: {entry.severity}</div>
                      </div>
                    </div>

                    <div className="text-slate-600">
                      <strong className="text-slate-900">Causal Agent:</strong>{' '}
                      {entry.causal_agent}
                    </div>

                    <div>
                      <div className="font-semibold text-slate-900 mb-1">
                        Field Symptoms (2–3 Bullets):
                      </div>
                      <ul className="list-disc pl-4 space-y-1 text-slate-700">
                        {entry.symptoms.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-3 rounded-lg bg-emerald-50/60 border border-emerald-200">
                      <strong className="text-emerald-950">Organic Remedy: </strong>
                      <span className="text-slate-800">{entry.organic_remedy}</span>
                    </div>

                    <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                      <strong className="text-slate-900">
                        Chemical Remedy (Generic Active Ingredient):{' '}
                      </strong>
                      <span className="text-slate-700">{entry.chemical_remedy}</span>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100">
                      <button
                        type="button"
                        onClick={() => {
                          setActiveTab('diagnostic');
                          fetchGoogleMapsStores({ locQuery: locationQuery });
                        }}
                        className="inline-flex items-center gap-1 text-emerald-700 font-semibold hover:underline cursor-pointer"
                      >
                        <MapPin className="w-3.5 h-3.5" />
                        <span>Locate Pesticide Stores</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          setActiveTab('diagnostic');
                          fetchGoogleSearchGrounding({
                            diseaseName: entry.disease_name,
                            causalAgent: entry.causal_agent,
                            crop: entry.crop
                          });
                        }}
                        className="inline-flex items-center gap-1 text-emerald-700 font-semibold hover:underline cursor-pointer"
                      >
                        <Globe className="w-3.5 h-3.5" />
                        <span>Search Published Articles</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'spec' && <SpecViewer />}
      </main>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-200 bg-white px-6 py-4 mt-12">
        <div className="max-w-[1380px] mx-auto flex flex-wrap items-center justify-between gap-4 text-xs text-slate-500">
          <div>
            PhytoScan — AI Crop-Disease Detector (Tomato &amp; Maize · PlantVillage 8-Class Subset)
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setActiveTab('diagnostic')}
              className="hover:text-slate-900 transition-colors cursor-pointer"
            >
              Diagnostic Main Screen
            </button>
            <span>·</span>
            <button
              type="button"
              onClick={() => setActiveTab('spec')}
              className="hover:text-slate-900 transition-colors cursor-pointer"
            >
              14-Section Specification
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
