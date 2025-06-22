// routes/predict.js
import axios from 'axios';

/**
 * POST /api/predict
 * Body: { image: "data:image/jpeg;base64,……" }
 */
export const predictImage = async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: 'No image provided' });

    /* ── 1. Strip the data-URL prefix and decode to bytes ───────────────── */
    const base64 = image.replace(/^data:image\/\w+;base64,/, '');
    const imgBuffer = Buffer.from(base64, 'base64');

    /* ── 2. Call the AWS API Gateway endpoint ──────────────────────────── */
    const { data: outer } = await axios.post(
      'https://lqq6925iw1.execute-api.us-east-1.amazonaws.com/production/predict-pneumonia',
      imgBuffer,
      { headers: { 'Content-Type': 'application/x-image' } }   // NOTE: capital “T”
    );

    /* outer looks like:
       {
         statusCode: 200,
         headers: {...},
         body: "{\"message\":\"Pneumonia detected (p=0.9969)\"}"
       }
    */

    /* ── 3. Unwrap the Lambda envelope ─────────────────────────────────── */
    const inner = JSON.parse(outer.body);     // ⇒ { message: "Pneumonia detected (p=0.9969)" }

    /* ── 4. Return just what the React client needs ────────────────────── */
    return res.json({ message: inner.message });   // e.g. "Pneumonia detected (p=0.9969)"
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Inference failed' });
  }
};