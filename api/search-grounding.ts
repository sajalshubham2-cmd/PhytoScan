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
    const { diseaseName, causalAgent, crop, customQuery } = req.body || {};

    if (!process.env.GEMINI_API_KEY) {
      res.status(500).json({
        error:
          'GEMINI_API_KEY is not configured. Add GEMINI_API_KEY in your Vercel Project Environment Variables.',
      });
      return;
    }

    const prompt = customQuery
      ? `Search for current agricultural extension articles, peer-reviewed plant pathology studies, and practical farmer management guidelines for: "${customQuery}" (Crop context: ${crop || 'Tomato / Maize'}). Summarize key findings, chemical/organic resistance management, and cite published agricultural sources.`
      : `Provide up-to-date agricultural extension details, recent management recommendations, weather/environmental risk triggers, and published articles for the crop disease "${diseaseName || 'Tomato Early Blight'}" caused by "${causalAgent || 'Alternaria solani'}" on ${crop || 'Tomato'}. Include specific guidance for smallholder farmers.`;

    const candidateModels = ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest'];
    let lastError: any = null;

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

        res.status(200).json({
          modelUsed: modelName,
          summary: text,
          articles,
          searchQueries,
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

    throw lastError || new Error('Failed to query Gemini Search Grounding');
  } catch (err: any) {
    res.status(500).json({
      error: err?.message || 'Failed to fetch Google Search grounded data.',
    });
  }
}
