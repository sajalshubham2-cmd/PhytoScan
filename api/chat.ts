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
      messages = [],
      tier = 'general',
      cropContext = 'Tomato',
      diseaseContext = 'Tomato Early Blight',
      remedyContext = 'Mancozeb 75% WP, Copper Oxychloride, Neem Oil',
    } = req.body || {};

    if (!process.env.GEMINI_API_KEY) {
      res.status(500).json({
        error:
          'GEMINI_API_KEY is not configured. Add GEMINI_API_KEY in your Vercel Project Environment Variables.',
      });
      return;
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      res.status(400).json({ error: 'Messages array is required.' });
      return;
    }

    const modelCandidatesByTier: Record<string, string[]> = {
      fast: ['gemini-3.1-flash-lite', 'gemini-3.8-flash', 'gemini-flash-latest'],
      general: ['gemini-3.5-flash', 'gemini-3.8-flash', 'gemini-flash-latest'],
      complex: ['gemini-3.1-pro-preview', 'gemini-3.8-flash', 'gemini-flash-latest'],
    };

    const candidates =
      modelCandidatesByTier[tier] || modelCandidatesByTier.general;

    const systemInstruction = `You are Dr. PhytoScan, a senior agronomist, plant pathologist, and crop medicine specialist assisting a farmer.
Current active field context:
- Selected Crop: ${cropContext}
- Current Diagnostic Result: ${diseaseContext}
- Recommended Active Ingredients: ${remedyContext}

Your responsibilities:
1. Recommend exact preventative and curative pesticides, fungicides, bio-fungicides, and plant medicines using generic active ingredient names (e.g., Mancozeb 75% WP, Chlorothalonil, Copper Oxychloride, Azoxystrobin, Propiconazole, Tebuconazole, Neem Oil, Trichoderma viride).
2. Always specify exact dilution rates (g/L or mL/L of water), spray interval (days), pre-harvest interval (PHI), personal protective equipment (PPE), and FRAC fungicide resistance rotation codes.
3. Keep answers clear, structured, and farmer-friendly.`;

    const history = messages.slice(0, -1).map((m: any) => ({
      role: m.role,
      parts: [{ text: m.text }],
    }));
    const latestMessage = messages[messages.length - 1]?.text || 'Hello';

    let lastError: any = null;
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

        res.status(200).json({
          modelUsed: modelName,
          reply: response.text || 'No response generated.',
        });
        return;
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
  } catch (err: any) {
    res.status(500).json({
      error: err?.message || 'Failed to generate chat response.',
    });
  }
}
