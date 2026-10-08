export type CropType = 'tomato' | 'maize';

export type DiagnosticStatus =
  | 'confident_disease'
  | 'confident_healthy'
  | 'low_confidence'
  | 'crop_mismatch';

export interface KnowledgeBaseEntry {
  index: number;
  crop: CropType;
  classKey: string;
  disease_name: string;
  causal_agent: string;
  severity: 'None' | 'Low' | 'Moderate' | 'High';
  symptoms: string[];
  organic_remedy: string;
  chemical_remedy: string;
  prevention_tips: string[];
  datasetAvailable: number;
  imagesUsed: number;
  trainCount: number;
  valCount: number;
  testCount: number;
  classWeight: number;
}

export interface DiagnosticHistoryItem {
  id: string;
  timestamp: string;
  cropSelected: CropType;
  sampleName: string;
  imagePreviewDataUrl: string;
  status: DiagnosticStatus;
  predictedClassKey: string | null;
  diseaseName: string;
  confidence: number;
  top2ClassKey: string | null;
  top2DiseaseName: string | null;
  top2Confidence: number;
  margin: number;
  rawCropMass: number;
  latencyMs: number;
  message: string;
  remedy: KnowledgeBaseEntry | null;
}

export interface SamplePreset {
  id: string;
  label: string;
  crop: CropType;
  description: string;
  category: 'standard' | 'edge_case';
  rawProbabilities: number[]; // length 8 matching KNOWLEDGE_BASE order
  svgDataUrl: string;
}

export function createLeafSvgDataUrl(opts: {
  bg: string;
  leafFill: string;
  veinStroke: string;
  lesionColor?: string;
  lesionType?: 'spots' | 'blight' | 'pustules' | 'stripes' | 'none' | 'blur';
  cropShape: 'tomato' | 'maize' | 'non_leaf';
  title: string;
}): string {
  const blurFilter =
    opts.lesionType === 'blur'
      ? `<filter id="b"><feGaussianBlur stdDeviation="5.5" /></filter>`
      : '';
  const groupFilter = opts.lesionType === 'blur' ? `filter="url(#b)"` : '';

  let bodySvg = '';
  if (opts.cropShape === 'non_leaf') {
    bodySvg = `
      <rect x="45" y="55" width="134" height="114" rx="14" fill="#64748B" stroke="#334155" stroke-width="3"/>
      <circle cx="112" cy="112" r="32" fill="#94A3B8" stroke="#1E293B" stroke-width="3"/>
      <line x1="65" y1="75" x2="159" y2="149" stroke="#CBD5E1" stroke-width="4"/>
    `;
  } else if (opts.cropShape === 'tomato') {
    bodySvg = `
      <path d="M112 22 C158 38, 188 84, 174 142 C162 178, 134 196, 112 204 C90 196, 62 178, 50 142 C36 84, 66 38, 112 22 Z" fill="${opts.leafFill}" stroke="${opts.veinStroke}" stroke-width="2.5"/>
      <path d="M112 26 L112 200" stroke="${opts.veinStroke}" stroke-width="2.5" stroke-linecap="round"/>
      <path d="M112 65 L154 48 M112 95 L166 76 M112 128 L162 112 M112 158 L146 146" stroke="${opts.veinStroke}" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M112 65 L70 48 M112 95 L58 76 M112 128 L62 112 M112 158 L78 146" stroke="${opts.veinStroke}" stroke-width="1.6" stroke-linecap="round"/>
    `;
  } else {
    // Maize blade
    bodySvg = `
      <path d="M36 196 C58 118, 104 52, 188 24 C174 108, 132 166, 68 202 Z" fill="${opts.leafFill}" stroke="${opts.veinStroke}" stroke-width="2.5"/>
      <path d="M50 198 C94 134, 136 78, 186 26" stroke="#D9F99D" stroke-width="3" stroke-linecap="round" fill="none"/>
      <path d="M62 184 C102 126, 142 78, 176 38" stroke="${opts.veinStroke}" stroke-width="1.2" stroke-dasharray="4 3" fill="none"/>
    `;
  }

  let lesionsSvg = '';
  if (opts.lesionType === 'spots' && opts.lesionColor) {
    lesionsSvg = `
      <circle cx="90" cy="86" r="10" fill="${opts.lesionColor}" opacity="0.88" stroke="#FEF08A" stroke-width="2"/>
      <circle cx="90" cy="86" r="5" fill="none" stroke="#451A03" stroke-width="1.2"/>
      <circle cx="136" cy="108" r="12" fill="${opts.lesionColor}" opacity="0.9" stroke="#FEF08A" stroke-width="2"/>
      <circle cx="136" cy="108" r="6" fill="none" stroke="#451A03" stroke-width="1.2"/>
      <circle cx="84" cy="138" r="8" fill="${opts.lesionColor}" opacity="0.85" stroke="#FEF08A" stroke-width="1.5"/>
      <circle cx="128" cy="68" r="7" fill="${opts.lesionColor}" opacity="0.85"/>
    `;
  } else if (opts.lesionType === 'blight' && opts.lesionColor) {
    lesionsSvg = `
      <path d="M112 22 C154 36, 182 76, 174 126 C148 132, 128 110, 118 82 C108 58, 102 40, 112 22 Z" fill="${opts.lesionColor}" opacity="0.88"/>
      <path d="M118 82 C128 110, 148 132, 172 126" fill="none" stroke="#E2E8F0" stroke-width="3" stroke-dasharray="3 2"/>
      <ellipse cx="78" cy="144" rx="18" ry="12" fill="${opts.lesionColor}" opacity="0.82"/>
    `;
  } else if (opts.lesionType === 'pustules' && opts.lesionColor) {
    lesionsSvg = `
      <circle cx="118" cy="88" r="4" fill="${opts.lesionColor}"/>
      <circle cx="128" cy="78" r="3.5" fill="${opts.lesionColor}"/>
      <circle cx="104" cy="112" r="4.5" fill="${opts.lesionColor}"/>
      <circle cx="94" cy="128" r="4" fill="${opts.lesionColor}"/>
      <circle cx="114" cy="134" r="3.5" fill="${opts.lesionColor}"/>
      <circle cx="82" cy="152" r="4.5" fill="${opts.lesionColor}"/>
      <circle cx="138" cy="96" r="4" fill="${opts.lesionColor}"/>
      <circle cx="148" cy="66" r="3.5" fill="${opts.lesionColor}"/>
    `;
  } else if (opts.lesionType === 'stripes' && opts.lesionColor) {
    lesionsSvg = `
      <rect x="86" y="102" width="38" height="7" rx="2" transform="rotate(-44 86 102)" fill="${opts.lesionColor}" opacity="0.9"/>
      <rect x="108" y="88" width="44" height="8" rx="2" transform="rotate(-44 108 88)" fill="${opts.lesionColor}" opacity="0.9"/>
      <rect x="68" y="138" width="34" height="7" rx="2" transform="rotate(-44 68 138)" fill="${opts.lesionColor}" opacity="0.9"/>
    `;
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 224 224" width="224" height="224">
    <defs>${blurFilter}</defs>
    <rect width="224" height="224" fill="${opts.bg}"/>
    <g ${groupFilter}>
      ${bodySvg}
      ${lesionsSvg}
    </g>
    <rect x="8" y="194" width="208" height="22" rx="4" fill="#0F172A" opacity="0.78"/>
    <text x="16" y="209" fill="#F8FAFC" font-family="monospace" font-size="10" font-weight="bold">${opts.title}</text>
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const KNOWLEDGE_BASE: KnowledgeBaseEntry[] = [
  {
    index: 0,
    crop: 'tomato',
    classKey: 'Tomato___Early_blight',
    disease_name: 'Tomato Early Blight',
    causal_agent: 'Alternaria solani (Fungus)',
    severity: 'Moderate',
    symptoms: [
      'Dark brown to black circular spots with concentric target-like rings on older lower leaves.',
      'Yellowing tissue around the dark spots that eventually causes lower leaves to dry and drop.',
      'Sunken dark leathery lesions near the stem end of developing tomato fruits.'
    ],
    organic_remedy:
      'Prune and burn infected lower leaves immediately. Spray Copper Oxychloride or Bordeaux mixture (3 g per liter of water) or neem oil extract (5 mL/L) every 7 to 10 days.',
    chemical_remedy:
      'Apply Mancozeb 75% WP (2.5 g/L of water) or Chlorothalonil 75% WP (2.0 g/L) or Azoxystrobin 23% SC (1.0 mL/L) as a foliar spray.',
    prevention_tips: [
      'Mulch around the base of plants with dry straw to stop soil-borne spores from splashing onto lower leaves.',
      'Use drip or furrow irrigation at soil level instead of overhead sprinklers.',
      'Rotate tomatoes with non-nightshade crops (maize or beans) for at least 2 seasons.'
    ],
    datasetAvailable: 1000,
    imagesUsed: 1000,
    trainCount: 700,
    valCount: 150,
    testCount: 150,
    classWeight: 0.9203
  },
  {
    index: 1,
    crop: 'tomato',
    classKey: 'Tomato___Late_blight',
    disease_name: 'Tomato Late Blight',
    causal_agent: 'Phytophthora infestans (Oomycete / Water Mold)',
    severity: 'High',
    symptoms: [
      'Large irregular pale-green to dark water-soaked patches starting at leaf tips and margins.',
      'Fine white cottony fungal growth on the underside of leaves during cool, wet mornings.',
      'Rapid browning and shriveling of entire stems and petioles within 3 to 5 days.'
    ],
    organic_remedy:
      'Immediately uproot and bury heavily infected plants away from the field. Spray preventative fixed copper fungicide (Copper Hydroxide 3.0 g/L) before rain events.',
    chemical_remedy:
      'Apply Cymoxanil 8% + Mancozeb 64% WP (2.5 g/L of water) or Metalaxyl-M 4% + Mancozeb 64% WP (2.5 g/L) thoroughly covering undersides of leaves.',
    prevention_tips: [
      'Space tomato plants 60 cm apart and stake them upright to maximize air circulation.',
      'Destroy volunteer tomato and potato plants near field borders before planting.',
      'Avoid evening watering so foliage stays completely dry overnight.'
    ],
    datasetAvailable: 1909,
    imagesUsed: 1000,
    trainCount: 700,
    valCount: 150,
    testCount: 150,
    classWeight: 0.9203
  },
  {
    index: 2,
    crop: 'tomato',
    classKey: 'Tomato___Septoria_leaf_spot',
    disease_name: 'Tomato Septoria Leaf Spot',
    causal_agent: 'Septoria lycopersici (Fungus)',
    severity: 'Moderate',
    symptoms: [
      'Numerous small circular spots (2 to 3 mm wide) with dark brown margins and pale gray or tan centers.',
      'Tiny black dot-like fruiting bodies (pycnidia) visible inside the gray center of each spot.',
      'Heavy defoliation progressing from the bottom of the plant upward, exposing fruits to sunscald.'
    ],
    organic_remedy:
      'Remove infected bottom leaves up to the first fruit cluster. Spray potassium bicarbonate solution (5 g/L) or copper soap fungicide every 7 days.',
    chemical_remedy:
      'Apply Chlorothalonil 75% WP (2.0 g/L of water) or Mancozeb 75% WP (2.5 g/L) at 10-day intervals.',
    prevention_tips: [
      'Wash hands and disinfect pruning shears with 70% rubbing alcohol between rows.',
      'Never walk through or cultivate the field while tomato foliage is wet from dew or rain.',
      'Plow under or burn all crop debris immediately after the final harvest.'
    ],
    datasetAvailable: 1771,
    imagesUsed: 1000,
    trainCount: 700,
    valCount: 150,
    testCount: 150,
    classWeight: 0.9203
  },
  {
    index: 3,
    crop: 'tomato',
    classKey: 'Tomato___healthy',
    disease_name: 'Healthy Tomato Leaf',
    causal_agent: 'None (Healthy Plant)',
    severity: 'None',
    symptoms: [
      'Uniform deep green leaf color with crisp, unblemished serrated edges.',
      'Clean leaf undersides free of spots, webbing, or powdery deposits.',
      'Strong turgid petioles and vigorous new shoot growth.'
    ],
    organic_remedy:
      'No curative treatment needed. Apply balanced compost tea or vermicompost top-dressing every 3 weeks to maintain plant vigor.',
    chemical_remedy:
      'No fungicide required. Continue standard balanced NPK fertilization according to crop growth stage.',
    prevention_tips: [
      'Inspect the lower canopy twice a week during humid or rainy weather.',
      'Maintain weed-free field borders to reduce insect vectors and fungal hosts.',
      'Keep lower leaves pruned 15 cm above the soil line once plants reach 60 cm tall.'
    ],
    datasetAvailable: 1591,
    imagesUsed: 1000,
    trainCount: 700,
    valCount: 150,
    testCount: 150,
    classWeight: 0.9203
  },
  {
    index: 4,
    crop: 'maize',
    classKey: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    disease_name: 'Maize Gray Leaf Spot',
    causal_agent: 'Cercospora zeae-maydis (Fungus)',
    severity: 'High',
    symptoms: [
      'Long, narrow, rectangular tan-to-gray lesions strictly bounded by the parallel leaf veins.',
      'Lesions display squared-off ends ("matchstick appearance") when held up to sunlight.',
      'Coalescing gray patches that blight and kill entire leaves during silking and grain fill.'
    ],
    organic_remedy:
      'Apply biological control spray containing Trichoderma harzianum or Bacillus subtilis (5 g/L of water) at early lesion appearance on lower leaves.',
    chemical_remedy:
      'Apply Propiconazole 25% EC (1.0 mL/L of water) or Azoxystrobin 18.2% + Difenoconazole 11.4% SC (1.0 mL/L) between tasseling and early silking.',
    prevention_tips: [
      'Practice deep tillage to bury infected maize stalks from the previous season.',
      'Rotate maize with soybean, cowpea, or groundnut for 1 to 2 years.',
      'Select locally adapted maize hybrids rated tolerant to Gray Leaf Spot.'
    ],
    datasetAvailable: 513,
    imagesUsed: 513,
    trainCount: 359,
    valCount: 77,
    testCount: 77,
    classWeight: 1.7945
  },
  {
    index: 5,
    crop: 'maize',
    classKey: 'Corn_(maize)___Common_rust_',
    disease_name: 'Maize Common Rust',
    causal_agent: 'Puccinia sorghi (Fungus)',
    severity: 'Moderate',
    symptoms: [
      'Small oval to elongated cinnamon-brown powdery pustules scattered on both upper and lower leaf surfaces.',
      'Outer skin (epidermis) of the leaf ruptures around each pustule, releasing brick-red spores when rubbed.',
      'Pustules turn dark brown to black as the plant matures toward harvest.'
    ],
    organic_remedy:
      'Spray wettable sulfur 80% WP (3.0 g/L of water) or neem oil emulsion (5 mL/L) early in the morning when rust pustules first appear on lower leaves.',
    chemical_remedy:
      'Apply Mancozeb 75% WP (2.5 g/L of water) or Tebuconazole 25.9% EC (1.0 mL/L) if pustules spread to the ear leaf before pollination.',
    prevention_tips: [
      'Plant early at the onset of the rainy season so crops mature before peak rust spore buildup.',
      'Eradicate wild Oxalis (wood sorrel) weeds around field margins, which act as the alternate host.',
      'Avoid excessive nitrogen top-dressing which produces soft, rust-susceptible leaf tissue.'
    ],
    datasetAvailable: 1192,
    imagesUsed: 1000,
    trainCount: 700,
    valCount: 150,
    testCount: 150,
    classWeight: 0.9203
  },
  {
    index: 6,
    crop: 'maize',
    classKey: 'Corn_(maize)___Northern_Leaf_Blight',
    disease_name: 'Maize Northern Leaf Blight',
    causal_agent: 'Exserohilum turcicum (Fungus)',
    severity: 'High',
    symptoms: [
      'Large cigar-shaped or elliptical grayish-green to tan lesions measuring 3 cm to 15 cm long.',
      'Lesions taper at both ends and cross seamlessly across leaf veins (unlike rectangular Gray Leaf Spot).',
      'Dusty dark olive-gray spore masses form in concentric zones inside lesions during damp weather.'
    ],
    organic_remedy:
      'Remove heavily blighted bottom leaves below the ear node and spray Copper Oxychloride (3.0 g/L) or Trichoderma viride bio-fungicide (5 g/L).',
    chemical_remedy:
      'Apply Mancozeb 75% WP (2.5 g/L of water) or Propiconazole 25% EC (1.0 mL/L) when lesions appear on the leaf below the primary ear.',
    prevention_tips: [
      'Ensure adequate potassium soil fertility, as potassium-deficient maize suffers severe blight.',
      'Compost or deeply plow maize stover after harvest to eliminate overwintering fungal spores.',
      'Maintain recommended plant population density (65,000–75,000 plants/ha) to keep the canopy ventilated.'
    ],
    datasetAvailable: 985,
    imagesUsed: 985,
    trainCount: 689,
    valCount: 148,
    testCount: 148,
    classWeight: 0.9344
  },
  {
    index: 7,
    crop: 'maize',
    classKey: 'Corn_(maize)___healthy',
    disease_name: 'Healthy Maize Leaf',
    causal_agent: 'None (Healthy Plant)',
    severity: 'None',
    symptoms: [
      'Broad, glossy dark-green leaf blades with prominent bright white central midrib.',
      'Absence of necrotic streaks, rust pustules, or chlorotic flecking.',
      'Whorl and ear-leaf surfaces intact and actively photosynthetic.'
    ],
    organic_remedy:
      'No curative treatment needed. Maintain soil moisture and apply organic mulch or farmyard manure between rows.',
    chemical_remedy:
      'No fungicide required. Follow standard split-application schedule for urea/nitrogen at knee-high and tasseling stages.',
    prevention_tips: [
      'Scout fields weekly in a W-pattern from knee-high stage through tasseling.',
      'Keep drainage channels clear to prevent waterlogging during heavy monsoon rains.',
      'Control stem borers and fall armyworm early so feeding wounds do not invite fungal infection.'
    ],
    datasetAvailable: 1162,
    imagesUsed: 1000,
    trainCount: 700,
    valCount: 150,
    testCount: 150,
    classWeight: 0.9203
  }
];

export const SAMPLE_PRESETS: SamplePreset[] = [
  {
    id: 'tomato_early_blight',
    label: 'Tomato — Early Blight',
    crop: 'tomato',
    description: 'Target-like concentric dark brown rings on lower tomato leaf.',
    category: 'standard',
    rawProbabilities: [0.914, 0.042, 0.028, 0.008, 0.002, 0.002, 0.002, 0.002],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#E2E8F0',
      leafFill: '#15803D',
      veinStroke: '#86EFAC',
      lesionColor: '#451A03',
      lesionType: 'spots',
      cropShape: 'tomato',
      title: 'Tomato___Early_blight'
    })
  },
  {
    id: 'tomato_late_blight',
    label: 'Tomato — Late Blight',
    crop: 'tomato',
    description: 'Large water-soaked dark necrotic patch with pale border.',
    category: 'standard',
    rawProbabilities: [0.031, 0.926, 0.022, 0.011, 0.003, 0.002, 0.003, 0.002],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#CBD5E1',
      leafFill: '#166534',
      veinStroke: '#4ADE80',
      lesionColor: '#1E293B',
      lesionType: 'blight',
      cropShape: 'tomato',
      title: 'Tomato___Late_blight'
    })
  },
  {
    id: 'tomato_septoria',
    label: 'Tomato — Septoria Spot',
    crop: 'tomato',
    description: 'Small circular spots with gray centers and dark margins.',
    category: 'standard',
    rawProbabilities: [0.048, 0.024, 0.896, 0.02, 0.003, 0.003, 0.003, 0.003],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#E2E8F0',
      leafFill: '#15803D',
      veinStroke: '#86EFAC',
      lesionColor: '#78350F',
      lesionType: 'pustules',
      cropShape: 'tomato',
      title: 'Tomato___Septoria_leaf_spot'
    })
  },
  {
    id: 'tomato_healthy',
    label: 'Tomato — Healthy Leaf',
    crop: 'tomato',
    description: 'Uniform vibrant green tomato leaflet with clean serrations.',
    category: 'standard',
    rawProbabilities: [0.006, 0.005, 0.007, 0.974, 0.002, 0.002, 0.002, 0.002],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#DCFCE7',
      leafFill: '#16A34A',
      veinStroke: '#BBF7D0',
      lesionType: 'none',
      cropShape: 'tomato',
      title: 'Tomato___healthy'
    })
  },
  {
    id: 'maize_gls',
    label: 'Maize — Gray Leaf Spot',
    crop: 'maize',
    description: 'Rectangular matchstick lesions bounded by parallel maize veins.',
    category: 'standard',
    rawProbabilities: [0.003, 0.002, 0.003, 0.002, 0.884, 0.032, 0.066, 0.008],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#E2E8F0',
      leafFill: '#15803D',
      veinStroke: '#86EFAC',
      lesionColor: '#78716C',
      lesionType: 'stripes',
      cropShape: 'maize',
      title: 'Corn___Gray_leaf_spot'
    })
  },
  {
    id: 'maize_common_rust',
    label: 'Maize — Common Rust',
    crop: 'maize',
    description: 'Cinnamon-brown powdery pustules scattered across maize blade.',
    category: 'standard',
    rawProbabilities: [0.002, 0.002, 0.002, 0.002, 0.014, 0.958, 0.014, 0.006],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#F1F5F9',
      leafFill: '#166534',
      veinStroke: '#86EFAC',
      lesionColor: '#B45309',
      lesionType: 'pustules',
      cropShape: 'maize',
      title: 'Corn___Common_rust_'
    })
  },
  {
    id: 'maize_nlb',
    label: 'Maize — Northern Leaf Blight',
    crop: 'maize',
    description: 'Large cigar-shaped tan lesions crossing maize veins.',
    category: 'standard',
    rawProbabilities: [0.002, 0.003, 0.002, 0.002, 0.058, 0.018, 0.905, 0.01],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#E2E8F0',
      leafFill: '#15803D',
      veinStroke: '#86EFAC',
      lesionColor: '#A16207',
      lesionType: 'blight',
      cropShape: 'maize',
      title: 'Corn___Northern_Leaf_Blight'
    })
  },
  {
    id: 'maize_healthy',
    label: 'Maize — Healthy Blade',
    crop: 'maize',
    description: 'Clean dark-green maize leaf blade with bright white midrib.',
    category: 'standard',
    rawProbabilities: [0.002, 0.002, 0.002, 0.002, 0.005, 0.004, 0.006, 0.977],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#ECFDF5',
      leafFill: '#16A34A',
      veinStroke: '#BBF7D0',
      lesionType: 'none',
      cropShape: 'maize',
      title: 'Corn___healthy'
    })
  },
  {
    id: 'edge_blurry_leaf',
    label: 'Edge Case — Blurry / Out-of-Focus Photo',
    crop: 'tomato',
    description: 'Triggers Low Confidence fallback (top1 < 0.65 or margin < 0.15).',
    category: 'edge_case',
    rawProbabilities: [0.34, 0.28, 0.21, 0.11, 0.02, 0.02, 0.01, 0.01],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#CBD5E1',
      leafFill: '#4D7C0F',
      veinStroke: '#84CC16',
      lesionColor: '#713F12',
      lesionType: 'blur',
      cropShape: 'tomato',
      title: 'Blurry_Out_Of_Focus.jpg'
    })
  },
  {
    id: 'edge_non_leaf',
    label: 'Edge Case — Non-Leaf / Soil & Tool Object',
    crop: 'tomato',
    description: 'Uniform diffuse logits across all classes -> Low Confidence fallback.',
    category: 'edge_case',
    rawProbabilities: [0.21, 0.19, 0.18, 0.17, 0.07, 0.06, 0.06, 0.06],
    svgDataUrl: createLeafSvgDataUrl({
      bg: '#E2E8F0',
      leafFill: '#64748B',
      veinStroke: '#334155',
      lesionType: 'none',
      cropShape: 'non_leaf',
      title: 'Tractor_Wrench_Object.jpg'
    })
  }
];

export const CROP_INDICES: Record<CropType, number[]> = {
  tomato: [0, 1, 2, 3],
  maize: [4, 5, 6, 7]
};

export interface InferenceResult {
  status: DiagnosticStatus;
  cropSelected: CropType;
  predictedClassKey: string | null;
  diseaseName: string;
  confidence: number;
  top2ClassKey: string | null;
  top2DiseaseName: string | null;
  top2Confidence: number;
  margin: number;
  rawCropMass: number;
  renormalizedProbs: { entry: KnowledgeBaseEntry; rawProb: number; renormProb: number }[];
  message: string;
  remedy: KnowledgeBaseEntry | null;
  latencyMs: number;
}

export function runCropAwareInference(
  rawProbs: number[],
  selectedCrop: CropType,
  tConf = 0.65,
  tMargin = 0.15,
  tCropMass = 0.35
): InferenceResult {
  const validIndices = CROP_INDICES[selectedCrop];
  const rawCropMass = validIndices.reduce((acc, idx) => acc + rawProbs[idx], 0);

  const renormalizedProbs = validIndices.map((idx) => {
    const rawProb = rawProbs[idx];
    const renormProb = rawCropMass > 1e-8 ? rawProb / rawCropMass : 0.25;
    return {
      entry: KNOWLEDGE_BASE[idx],
      rawProb,
      renormProb
    };
  });

  if (rawCropMass < tCropMass) {
    const otherCropName = selectedCrop === 'tomato' ? 'Maize' : 'Tomato';
    return {
      status: 'crop_mismatch',
      cropSelected: selectedCrop,
      predictedClassKey: null,
      diseaseName: 'Crop Mismatch Detected',
      confidence: Number(rawCropMass.toFixed(4)),
      top2ClassKey: null,
      top2DiseaseName: null,
      top2Confidence: 0,
      margin: 0,
      rawCropMass: Number(rawCropMass.toFixed(4)),
      renormalizedProbs,
      message: `The uploaded image does not appear to be a ${
        selectedCrop === 'tomato' ? 'Tomato' : 'Maize'
      } leaf (only ${(rawCropMass * 100).toFixed(
        1
      )}% probability mass on ${selectedCrop}; likely ${otherCropName}). Please verify the selected crop or retake a clear close-up photo of a single leaf.`,
      remedy: null,
      latencyMs: 38.4
    };
  }

  const sorted = [...renormalizedProbs].sort((a, b) => b.renormProb - a.renormProb);
  const top1 = sorted[0];
  const top2 = sorted[1];
  const p1 = top1.renormProb;
  const p2 = top2.renormProb;
  const margin = p1 - p2;

  if (p1 < tConf || margin < tMargin) {
    return {
      status: 'low_confidence',
      cropSelected: selectedCrop,
      predictedClassKey: null,
      diseaseName: 'No Disease Confidently Detected',
      confidence: Number(p1.toFixed(4)),
      top2ClassKey: top2.entry.classKey,
      top2DiseaseName: top2.entry.disease_name,
      top2Confidence: Number(p2.toFixed(4)),
      margin: Number(margin.toFixed(4)),
      rawCropMass: Number(rawCropMass.toFixed(4)),
      renormalizedProbs,
      message:
        'No disease confidently detected. Please retake the photo in bright natural daylight with a single leaf centered and in sharp focus.',
      remedy: null,
      latencyMs: 41.2
    };
  }

  const isHealthy = top1.entry.classKey.endsWith('healthy');
  const status: DiagnosticStatus = isHealthy ? 'confident_healthy' : 'confident_disease';

  const message = isHealthy
    ? `Healthy ${selectedCrop === 'tomato' ? 'Tomato' : 'Maize'} leaf detected (${(
        p1 * 100
      ).toFixed(1)}% confidence). No disease symptoms found.`
    : `Detected ${top1.entry.disease_name} with ${(p1 * 100).toFixed(
        1
      )}% confidence. Review recommended remedies below.`;

  return {
    status,
    cropSelected: selectedCrop,
    predictedClassKey: top1.entry.classKey,
    diseaseName: top1.entry.disease_name,
    confidence: Number(p1.toFixed(4)),
    top2ClassKey: top2.entry.classKey,
    top2DiseaseName: top2.entry.disease_name,
    top2Confidence: Number(p2.toFixed(4)),
    margin: Number(margin.toFixed(4)),
    rawCropMass: Number(rawCropMass.toFixed(4)),
    renormalizedProbs,
    message,
    remedy: top1.entry,
    latencyMs: 39.6
  };
}
