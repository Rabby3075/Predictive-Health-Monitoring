import React, { useEffect, useState } from "react";
import './App.css';
import ModelForm from "./components/ModelForm";

function App() {
  const [models, setModels] = useState([]);
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/models")
      .then((res) => res.json())
      .then((data) => {
        setModels(data.models);
        setFeatures(data.features);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20 }}>
      <h2>Predictive Health Monitoring</h2>
      <ModelForm models={models} features={features} />
    </div>
  );
}

export default App;
