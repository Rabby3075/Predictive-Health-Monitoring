import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  TextField,
  Typography
} from "@mui/material";
import React, { useEffect, useState } from "react";

// Unique values for dropdowns (from your screenshots)
const AGE_OPTIONS = ["[60-70)", "[70-80)", "[80-90)", "[90-100)"];
const MEDICAL_SPECIALTY_OPTIONS = [
  "Missing", "Other", "InternalMedicine", "Family/GeneralPractice", "Cardiology", "Surgery", "Emergency/Trauma"
];
const DIAG_OPTIONS = [
  "Circulatory", "Other", "Injury", "Digestive", "Respiratory", "Diabetes", "Musculoskeletal", "Missing"
];
const GLUCOSE_OPTIONS = ["no", "normal", "high"];
const A1C_OPTIONS = ["no", "normal", "high"];
const CHANGE_OPTIONS = ["no", "yes"];
const DIABETES_MED_OPTIONS = ["yes", "no"];

const NUMERIC_FIELDS = [
  "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications", "n_outpatient", "n_inpatient", "n_emergency"
];

const RAW_FIELDS = [
  "age", ...NUMERIC_FIELDS,
  "medical_specialty", "diag_1", "diag_2", "diag_3",
  "glucose_test", "A1Ctest", "change", "diabetes_med"
];

const AGE_MAP = {
  "[60-70)": 65,
  "[70-80)": 75,
  "[80-90)": 85,
  "[90-100)": 95
};

const CATEGORICAL_COLUMNS = [
  "medical_specialty", "diag_1", "diag_2", "diag_3", "glucose_test", "A1Ctest", "change", "diabetes_med", "primary_diagnosis"
];

// All possible one-hot columns (from your preprocessing)
const ONE_HOT_MAP = {
  medical_specialty: MEDICAL_SPECIALTY_OPTIONS.map(v => v === "Missing" ? "Unknown" : v),
  diag_1: DIAG_OPTIONS,
  diag_2: DIAG_OPTIONS,
  diag_3: DIAG_OPTIONS,
  glucose_test: GLUCOSE_OPTIONS,
  A1Ctest: A1C_OPTIONS,
  change: CHANGE_OPTIONS,
  diabetes_med: DIABETES_MED_OPTIONS,
  primary_diagnosis: DIAG_OPTIONS,
};

const API_URL = "https://predictive-health-monitoring.onrender.com";

function getPrimaryDiagnosis(diag1, diag2, diag3) {
  const counts = {};
  [diag1, diag2, diag3].forEach(d => {
    counts[d] = (counts[d] || 0) + 1;
  });
  // Return the most frequent
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

function ModelForm({ features }) {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [input, setInput] = useState({
    age: AGE_OPTIONS[0],
    time_in_hospital: "",
    n_lab_procedures: "",
    n_procedures: "",
    n_medications: "",
    n_outpatient: "",
    n_inpatient: "",
    n_emergency: "",
    medical_specialty: MEDICAL_SPECIALTY_OPTIONS[0],
    diag_1: DIAG_OPTIONS[0],
    diag_2: DIAG_OPTIONS[0],
    diag_3: DIAG_OPTIONS[0],
    glucose_test: GLUCOSE_OPTIONS[0],
    A1Ctest: A1C_OPTIONS[0],
    change: CHANGE_OPTIONS[0],
    diabetes_med: DIABETES_MED_OPTIONS[0],
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  const handleChange = (e) => {
    setInput({ ...input, [e.target.name]: e.target.value });
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    setResult(null);
    setError("");
  };

  // Preprocess user input to match model input
  function preprocessInput(raw) {
    // 1. Map age
    const processed = { ...raw };
    processed.age = AGE_MAP[raw.age];
    // 2. Replace 'Missing' with 'Unknown' for medical_specialty
    processed.medical_specialty = raw.medical_specialty === "Missing" ? "Unknown" : raw.medical_specialty;
    // 3. Compute primary_diagnosis
    processed.primary_diagnosis = getPrimaryDiagnosis(raw.diag_1, raw.diag_2, raw.diag_3);
    // 4. One-hot encode all categorical columns
    let oneHot = {};
    for (const col of CATEGORICAL_COLUMNS) {
      for (const val of ONE_HOT_MAP[col]) {
        const key = `${col}_${val}`;
        oneHot[key] = (processed[col] === val) ? 1 : 0;
      }
    }
    // 5. Collect all features in the order expected by the model
    let modelInput = {
      age: processed.age,
      time_in_hospital: Number(raw.time_in_hospital),
      n_lab_procedures: Number(raw.n_lab_procedures),
      n_procedures: Number(raw.n_procedures),
      n_medications: Number(raw.n_medications),
      n_outpatient: Number(raw.n_outpatient),
      n_inpatient: Number(raw.n_inpatient),
      n_emergency: Number(raw.n_emergency),
      ...oneHot
    };
    return modelInput;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError("");
    setLoading(true);
    try {
      const processed = preprocessInput(input);
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          features: processed,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
        setOpen(true);
      } else {
        setError(data.detail || "Prediction failed.");
      }
    } catch (err) {
      setError("Server error.");
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => setOpen(false);

  useEffect(() => {
    const fetchModels = async () => {
      setLoadingModels(true);
      try {
        const res = await fetch(`${API_URL}/models`);
        const data = await res.json();
        setModels(data.models);
      } catch (err) {
        setError("Failed to load models");
      }
      setLoadingModels(false);
    };
    fetchModels();
  }, []);

  return (
    <Paper elevation={3} sx={{ p: 4, mt: 4, borderRadius: 3 }}>
      <form onSubmit={handleSubmit}>
        <Typography variant="h5" gutterBottom>Model Selection</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Model</InputLabel>
          <Select value={selectedModel} label="Model" onChange={handleModelChange}>
            {models.map((m) => (
              <MenuItem key={m} value={m}>
                {m.replace(/_/g, " ")}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Divider sx={{ my: 2 }} />
        <Typography variant="h6" gutterBottom>Age</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Age</InputLabel>
          <Select name="age" value={input.age} label="Age" onChange={handleChange}>
            {AGE_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Numerical Features</Typography>
        <Grid container spacing={2}>
          {NUMERIC_FIELDS.map((f) => (
            <Grid item xs={12} sm={6} md={4} key={f}>
              <TextField
                label={f.replace(/_/g, " ")}
                name={f}
                value={input[f]}
                onChange={handleChange}
                type="number"
                fullWidth
                required
                size="small"
                variant="outlined"
                InputProps={{ inputProps: { min: 0 } }}
                placeholder={`Enter ${f.replace(/_/g, " ")}`}
              />
            </Grid>
          ))}
        </Grid>
        <Divider sx={{ my: 2 }} />
        <Typography variant="h6" gutterBottom>Medical Specialty</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Medical Specialty</InputLabel>
          <Select name="medical_specialty" value={input.medical_specialty} label="Medical Specialty" onChange={handleChange}>
            {MEDICAL_SPECIALTY_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Diagnosis 1</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Diagnosis 1</InputLabel>
          <Select name="diag_1" value={input.diag_1} label="Diagnosis 1" onChange={handleChange}>
            {DIAG_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Diagnosis 2</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Diagnosis 2</InputLabel>
          <Select name="diag_2" value={input.diag_2} label="Diagnosis 2" onChange={handleChange}>
            {DIAG_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Diagnosis 3</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Diagnosis 3</InputLabel>
          <Select name="diag_3" value={input.diag_3} label="Diagnosis 3" onChange={handleChange}>
            {DIAG_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Glucose Test</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Glucose Test</InputLabel>
          <Select name="glucose_test" value={input.glucose_test} label="Glucose Test" onChange={handleChange}>
            {GLUCOSE_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>A1C Test</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>A1C Test</InputLabel>
          <Select name="A1Ctest" value={input.A1Ctest} label="A1C Test" onChange={handleChange}>
            {A1C_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Change</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Change</InputLabel>
          <Select name="change" value={input.change} label="Change" onChange={handleChange}>
            {CHANGE_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="h6" gutterBottom>Diabetes Medication</Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Diabetes Medication</InputLabel>
          <Select name="diabetes_med" value={input.diabetes_med} label="Diabetes Medication" onChange={handleChange}>
            {DIABETES_MED_OPTIONS.map((opt) => (
              <MenuItem key={opt} value={opt}>{opt}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          sx={{ mt: 4, fontWeight: "bold", fontSize: 16, borderRadius: 2 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} color="inherit" /> : "Predict"}
        </Button>
        {error && (
          <Alert severity="error" sx={{ mt: 4 }}>
            <strong>Error:</strong> {error}
          </Alert>
        )}
      </form>
      {/* Modal for result */}
      <Dialog open={open} onClose={handleClose} maxWidth="xs" fullWidth>
        <DialogTitle>Prediction Result</DialogTitle>
        <DialogContent>
          {result && (
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" py={2}>
              <Typography
                variant="h6"
                gutterBottom
                color={result.prediction === 1 ? "error" : "success"}
                fontWeight="bold"
              >
                {result.prediction === 1 ? "Readmitted" : "Not Readmitted"}
              </Typography>
              <Box position="relative" display="inline-flex" my={2}>
                <CircularProgress
                  variant="determinate"
                  value={result.probability * 100}
                  size={120}
                  thickness={5}
                  color={result.prediction === 1 ? "error" : "success"}
                />
                <Box
                  top={0}
                  left={0}
                  bottom={0}
                  right={0}
                  position="absolute"
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                >
                  <Typography variant="h5" component="div" color="textPrimary">
                    {(result.probability * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
              <Button onClick={handleClose} variant="outlined" sx={{ mt: 2 }}>
                Close
              </Button>
            </Box>
          )}
        </DialogContent>
      </Dialog>
    </Paper>
  );
}

export default ModelForm; 