import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { corsHeaders } from '../_shared/cors.ts'

const MODEL_FEATURES = [
  'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
  'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
  'age', 'gender', 'jaundice', 'autism'
];

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { age, gender, jaundice, autism, behavioralScores } = await req.json()

    // Convert form data to model features
    const features = {
      age,
      gender: gender === 'm' ? 1 : 0,
      jaundice: jaundice === 'yes' ? 1 : 0,
      autism: autism === 'yes' ? 1 : 0,
      ...Object.entries(behavioralScores).reduce((acc, [key, value]) => ({
        ...acc,
        [`${key}_Score`]: value === '1' ? 1 : 0
      }), {})
    }

    // TODO: Load and use your trained model here
    // For now, returning mock prediction
    const mockPrediction = {
      result: features.autism === 1 || 
              Object.values(behavioralScores).filter(v => v === '1').length >= 6,
      probability: 0.85
    }

    return new Response(
      JSON.stringify(mockPrediction),
      { 
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        }
      }
    )
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 400,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        }
      }
    )
  }
})