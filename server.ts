import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

async function generateWithSearchGrounding(prompt: string) {
  const candidateModels = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest'];
  let lastError: unknown = null;

  for (const modelName of candidateModels) {
    try {
      const response = await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          systemInstruction:
            'You are a senior agricultural extension plant pathologist. Provide concise, practical, farmer-friendly insights backed by current university extension publications, FAO/CABI crop protection sheets, and peer-reviewed agronomy research. Structure your response with: 1) Latest Research & Field Insights, 2) Integrated Pest Management (IPM) & Fungicide Resistance Notes, and 3) Key Takeaways from Published Extension Articles.',
          tools: [{ googleSearch: {} }],
        },
      });

      const text = response.text || 'No summary returned.';
      const rawChunks =
        response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
      const searchQueries =
        response.candidates?.[0]?.groundingMetadata?.webSearchQueries || [];

      const articles: { title: string; uri: string }[] = [];
      const seenUris = new Set<string>();

      for (const chunk of rawChunks) {
        const uri = chunk.web?.uri;
        const title = chunk.web?.title || uri || 'Published Agricultural Resource';
        if (uri && !seenUris.has(uri)) {
          seenUris.add(uri);
          articles.push({ title, uri });
        }
      }

      return {
        modelUsed: modelName,
        summary: text,
        articles,
        searchQueries,
      };
    } catch (err: any) {
      lastError = err;
      const status = err?.status || err?.code;
      const msg = String(err?.message || '');
      if (status === 404 || msg.includes('404') || msg.includes('NOT_FOUND')) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error('Failed to query Gemini Search Grounding');
}

async function generateWithMapsGrounding(opts: {
  diseaseName: string;
  causalAgent: string;
  crop: string;
  chemicalRemedy: string;
  organicRemedy: string;
  locationQuery: string;
  latitude?: number;
  longitude?: number;
}) {
  const candidateModels = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest'];
  let lastError: unknown = null;

  const hasCoords =
    typeof opts.latitude === 'number' &&
    !Number.isNaN(opts.latitude) &&
    typeof opts.longitude === 'number' &&
    !Number.isNaN(opts.longitude);

  const locationContext = opts.locationQuery
    ? `in or near ${opts.locationQuery}`
    : hasCoords
    ? `near coordinates (${opts.latitude?.toFixed(4)}, ${opts.longitude?.toFixed(4)})`
    : 'nearby';

  const prompt = `A farmer is treating or preventing "${opts.diseaseName}" (${opts.causalAgent}) on ${opts.crop}.
1. First, list the exact specific pesticides, fungicides, and organic plant medicines required to prevent and control "${opts.diseaseName}" (specifically referencing active ingredients such as: ${opts.chemicalRemedy}; and organic options: ${opts.organicRemedy}), including protective equipment, dosage mixing ratio, and spray timing.
2. Second, find real agricultural supply stores, agro-chemical dealers, farm co-ops, garden centers, or plant pharmacies ${locationContext} where the farmer can purchase these fungicides, copper sprays, neem oil, and crop protection medicines. Provide details on each store found.`;

  for (const modelName of candidateModels) {
    try {
      const config: Record<string, any> = {
        tools: [{ googleMaps: {} }],
      };

      if (hasCoords) {
        config.toolConfig = {
          retrievalConfig: {
            latLng: {
              latitude: Number(opts.latitude),
              longitude: Number(opts.longitude),
            },
          },
        };
      }

      const response = await ai.models.generateContent({
        model: modelName,
        contents: prompt,
        config,
      });

      const text = response.text || 'No store recommendations returned.';
      const rawChunks =
        response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];

      const places: {
        title: string;
        uri: string;
        reviewSnippets: string[];
      }[] = [];
      const seenUris = new Set<string>();

      for (const chunk of rawChunks as any[]) {
        const mapsObj = chunk.maps;
        if (mapsObj && mapsObj.uri) {
          if (!seenUris.has(mapsObj.uri)) {
            seenUris.add(mapsObj.uri);
            const rawSnippets =
              mapsObj.placeAnswerSources?.reviewSnippets || [];
            const reviewSnippets: string[] = [];
            for (const s of rawSnippets) {
              if (typeof s === 'string') {
                reviewSnippets.push(s);
              } else if (s && typeof s.text === 'string') {
                reviewSnippets.push(s.text);
              } else if (s && typeof s.reviewText === 'string') {
                reviewSnippets.push(s.reviewText);
              }
            }
            places.push({
              title: mapsObj.title || 'Agricultural Supply Store on Google Maps',
              uri: mapsObj.uri,
              reviewSnippets,
            });
          }
        }
      }

      return {
        modelUsed: modelName,
        guidance: text,
        places,
        locationUsed: opts.locationQuery || (hasCoords ? `${opts.latitude?.toFixed(4)}, ${opts.longitude?.toFixed(4)}` : 'Default Region'),
      };
    } catch (err: any) {
      lastError = err;
      const status = err?.status || err?.code;
      const msg = String(err?.message || '');
      if (status === 404 || msg.includes('404') || msg.includes('NOT_FOUND')) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error('Failed to query Gemini Maps Grounding');
}

async function runMultiTurnChat(opts: {
  messages: { role: 'user' | 'model'; text: string }[];
  tier: 'fast' | 'general' | 'complex';
  cropContext: string;
  diseaseContext: string;
  remedyContext: string;
}) {
  const modelCandidatesByTier: Record<'fast' | 'general' | 'complex', string[]> = {
    fast: ['gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-flash-latest'],
    general: ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest'],
    complex: ['gemini-3.1-pro-preview', 'gemini-3.8-flash', 'gemini-flash-latest'],
  };

  const candidates = modelCandidatesByTier[opts.tier] || modelCandidatesByTier.general;
  const systemInstruction = `You are Dr. PhytoScan, a senior agronomist, plant pathologist, and crop medicine specialist assisting a farmer.
Current active field context:
- Selected Crop: ${opts.cropContext}
- Current Diagnostic Result: ${opts.diseaseContext}
- Recommended Active Ingredients: ${opts.remedyContext}

Your responsibilities:
1. Recommend exact preventative and curative pesticides, fungicides, bio-fungicides, and plant medicines using generic active ingredient names (e.g., Mancozeb 75% WP, Chlorothalonil, Copper Oxychloride, Azoxystrobin, Propiconazole, Tebuconazole, Neem Oil, Trichoderma viride).
2. Always specify exact dilution rates (g/L or mL/L of water), spray interval (days), pre-harvest interval (PHI), personal protective equipment (PPE), and FRAC fungicide resistance rotation codes.
3. Keep answers clear, structured, and farmer-friendly.`;

  const history = opts.messages.slice(0, -1).map((m) => ({
    role: m.role,
    parts: [{ text: m.text }],
  }));
  const latestMessage =
    opts.messages[opts.messages.length - 1]?.text || 'Hello';

  let lastError: unknown = null;
  for (const modelName of candidates) {
    try {
      const chat = ai.chats.create({
        model: modelName,
        history,
        config: {
          systemInstruction,
        },
      });

      const response = await chat.sendMessage({
        message: latestMessage,
      });

      return {
        modelUsed: modelName,
        reply: response.text || 'No response generated.',
      };
    } catch (err: any) {
      lastError = err;
      const status = err?.status || err?.code;
      const msg = String(err?.message || '');
      if (
        status === 404 ||
        status === 403 ||
        status === 429 ||
        msg.includes('404') ||
        msg.includes('403') ||
        msg.includes('429') ||
        msg.includes('NOT_FOUND')
      ) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error('Failed to generate chat response.');
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '6mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', service: 'phytoscan-backend' });
  });

  app.post('/api/search-grounding', async (req, res) => {
    try {
      const { diseaseName, causalAgent, crop, customQuery } = req.body || {};

      if (!process.env.GEMINI_API_KEY) {
        res.status(500).json({
          error:
            'GEMINI_API_KEY is not configured. Please check the Settings > Secrets panel.',
        });
        return;
      }

      const queryPrompt = customQuery
        ? `Search for current agricultural extension articles, peer-reviewed plant pathology studies, and practical farmer management guidelines for: "${customQuery}" (Crop context: ${crop || 'Tomato / Maize'}). Summarize key findings, chemical/organic resistance management, and cite published agricultural sources.`
        : `Provide up-to-date agricultural extension details, recent management recommendations, weather/environmental risk triggers, and published articles for the crop disease "${diseaseName || 'Tomato Early Blight'}" caused by "${causalAgent || 'Alternaria solani'}" on ${crop || 'Tomato'}. Include specific guidance for smallholder farmers.`;

      const result = await generateWithSearchGrounding(queryPrompt);
      res.json(result);
    } catch (err: any) {
      const message = err?.message || 'Failed to fetch Google Search grounded data.';
      const status = err?.status === 403 || err?.status === 400 ? err.status : 500;
      res.status(status).json({ error: message });
    }
  });

  app.post('/api/maps-grounding', async (req, res) => {
    try {
      const {
        diseaseName,
        causalAgent,
        crop,
        chemicalRemedy,
        organicRemedy,
        locationQuery,
        latitude,
        longitude,
      } = req.body || {};

      if (!process.env.GEMINI_API_KEY) {
        res.status(500).json({
          error:
            'GEMINI_API_KEY is not configured. Please check the Settings > Secrets panel.',
        });
        return;
      }

      const result = await generateWithMapsGrounding({
        diseaseName: diseaseName || 'Tomato Early Blight',
        causalAgent: causalAgent || 'Alternaria solani',
        crop: crop || 'Tomato',
        chemicalRemedy:
          chemicalRemedy || 'Mancozeb 75% WP (2.5 g/L) or Chlorothalonil 75% WP',
        organicRemedy:
          organicRemedy || 'Copper Oxychloride (3 g/L) or Neem Oil extract',
        locationQuery: locationQuery || '',
        latitude: typeof latitude === 'number' ? latitude : undefined,
        longitude: typeof longitude === 'number' ? longitude : undefined,
      });

      res.json(result);
    } catch (err: any) {
      const message =
        err?.message || 'Failed to fetch Google Maps grounded store locations.';
      const status = err?.status === 403 || err?.status === 400 ? err.status : 500;
      res.status(status).json({ error: message });
    }
  });

  app.post('/api/chat', async (req, res) => {
    try {
      const { messages, tier, cropContext, diseaseContext, remedyContext } =
        req.body || {};

      if (!process.env.GEMINI_API_KEY) {
        res.status(500).json({
          error:
            'GEMINI_API_KEY is not configured. Please check the Settings > Secrets panel.',
        });
        return;
      }

      if (!Array.isArray(messages) || messages.length === 0) {
        res.status(400).json({ error: 'Messages array is required.' });
        return;
      }

      const result = await runMultiTurnChat({
        messages,
        tier: tier || 'general',
        cropContext: cropContext || 'Tomato',
        diseaseContext: diseaseContext || 'Tomato Early Blight',
        remedyContext:
          remedyContext || 'Mancozeb 75% WP, Copper Oxychloride, Neem Oil',
      });

      res.json(result);
    } catch (err: any) {
      const message = err?.message || 'Failed to generate chat response.';
      const status = err?.status === 403 || err?.status === 400 ? err.status : 500;
      res.status(status).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, '0.0.0.0', () => {
    console.log(`PhytoScan server running on http://0.0.0.0:${port}`);
  });
}

startServer();
