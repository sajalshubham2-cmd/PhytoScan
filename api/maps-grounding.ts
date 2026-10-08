import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method Not Allowed' });
    return;
  }

  try {
    const {
      diseaseName = 'Tomato Early Blight',
      causalAgent = 'Alternaria solani',
      crop = 'Tomato',
      chemicalRemedy = 'Mancozeb 75% WP (2.5 g/L) or Chlorothalonil 75% WP',
      organicRemedy = 'Copper Oxychloride (3 g/L) or Neem Oil extract',
      locationQuery = '',
      latitude,
      longitude,
    } = req.body || {};

    if (!process.env.GEMINI_API_KEY) {
      res.status(500).json({
        error:
          'GEMINI_API_KEY is not configured. Add GEMINI_API_KEY in your Vercel Project Environment Variables.',
      });
      return;
    }

    const hasCoords =
      typeof latitude === 'number' &&
      !Number.isNaN(latitude) &&
      typeof longitude === 'number' &&
      !Number.isNaN(longitude);

    const locationContext = locationQuery
      ? `in or near ${locationQuery}`
      : hasCoords
      ? `near coordinates (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`
      : 'nearby';

    const prompt = `A farmer is treating or preventing "${diseaseName}" (${causalAgent}) on ${crop}.
1. First, list the exact specific pesticides, fungicides, and organic plant medicines required to prevent and control "${diseaseName}" (specifically referencing active ingredients such as: ${chemicalRemedy}; and organic options: ${organicRemedy}), including protective equipment, dosage mixing ratio, and spray timing.
2. Second, find real agricultural supply stores, agro-chemical dealers, farm co-ops, garden centers, or plant pharmacies ${locationContext} where the farmer can purchase these fungicides, copper sprays, neem oil, and crop protection medicines. Provide details on each store found.`;

    const candidateModels = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest'];
    let lastError: any = null;

    for (const modelName of candidateModels) {
      try {
        const config: Record<string, any> = {
          tools: [{ googleMaps: {} }],
        };

        if (hasCoords) {
          config.toolConfig = {
            retrievalConfig: {
              latLng: {
                latitude: Number(latitude),
                longitude: Number(longitude),
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
          if (mapsObj && mapsObj.uri && !seenUris.has(mapsObj.uri)) {
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

        res.status(200).json({
          modelUsed: modelName,
          guidance: text,
          places,
          locationUsed:
            locationQuery ||
            (hasCoords
              ? `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`
              : 'Default Region'),
        });
        return;
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
  } catch (err: any) {
    res.status(500).json({
      error:
        err?.message || 'Failed to fetch Google Maps grounded store locations.',
    });
  }
}
