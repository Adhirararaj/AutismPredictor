import React, { useState } from 'react';
import { Brain, AlertCircle, Loader2 } from 'lucide-react';

interface FormData {
  age: number;
  gender: string;
  ethnicity: string;
  jaundice: string;
  autism: string;
  country: string;
  usedAppBefore: string;
  relation: string;
  behavioralScores: {
    [key: string]: string;
  };
}

const initialFormData: FormData = {
  age: 0,
  gender: '',
  ethnicity: '',
  jaundice: '',
  autism: '',
  country: '',
  usedAppBefore: '',
  relation: '',
  behavioralScores: {
    A1: '', A2: '', A3: '', A4: '', A5: '',
    A6: '', A7: '', A8: '', A9: '', A10: ''
  }
};

function App() {
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{ result: boolean; probability: number } | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    if (name.startsWith('A')) {
      setFormData(prev => ({
        ...prev,
        behavioralScores: {
          ...prev.behavioralScores,
          [name]: value
        }
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/predict`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-purple-900 text-white py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-center gap-3 mb-8">
          <Brain className="w-12 h-12 text-blue-400" />
          <h1 className="text-4xl font-bold">Autism Spectrum Predictor</h1>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-8 shadow-xl border border-white/20">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="age" className="block text-sm font-medium mb-2">
                  Age
                </label>
                <input
                  type="number"
                  id="age"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label htmlFor="gender" className="block text-sm font-medium mb-2">
                  Gender
                </label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Select gender</option>
                  <option value="m">Male</option>
                  <option value="f">Female</option>
                </select>
              </div>
            </div>

            {/* Behavioral Scores */}
            <div className="space-y-4">
              <h2 className="text-xl font-semibold">Behavioral Assessment</h2>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.keys(formData.behavioralScores).map((key) => (
                  <div key={key}>
                    <label htmlFor={key} className="block text-sm font-medium mb-2">
                      {key}
                    </label>
                    <select
                      id={key}
                      name={key}
                      value={formData.behavioralScores[key]}
                      onChange={handleInputChange}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required
                    >
                      <option value="">Select</option>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {/* Additional Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="jaundice" className="block text-sm font-medium mb-2">
                  Born with Jaundice
                </label>
                <select
                  id="jaundice"
                  name="jaundice"
                  value={formData.jaundice}
                  onChange={handleInputChange}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Select option</option>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>

              <div>
                <label htmlFor="autism" className="block text-sm font-medium mb-2">
                  Family Member with Autism
                </label>
                <select
                  id="autism"
                  name="autism"
                  value={formData.autism}
                  onChange={handleInputChange}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Select option</option>
                  <option value="yes">Yes</option>
                  <option value="no">No</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </>
              ) : (
                'Get Prediction'
              )}
            </button>
          </form>

          {error && (
            <div className="mt-6 bg-red-500/20 border border-red-500/50 rounded-lg p-4 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-red-200">{error}</p>
            </div>
          )}

          {prediction && (
            <div className="mt-6 bg-blue-500/20 border border-blue-500/50 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-2">Prediction Result</h3>
              <p className="text-lg">
                The model predicts that the individual
                <span className={`font-bold ${prediction.result ? 'text-yellow-400' : 'text-green-400'}`}>
                  {prediction.result ? ' may ' : ' may not '}
                </span>
                be on the autism spectrum.
              </p>
              <p className="mt-2 text-blue-200">
                Confidence: {(prediction.probability * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;