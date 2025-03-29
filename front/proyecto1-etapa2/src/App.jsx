import React, { useState, useRef } from "react";
import { FaKeyboard, FaFileAlt, FaCogs, FaFolderOpen } from "react-icons/fa";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [activeTab, setActiveTab] = useState("unico");

  const fileInputRef1 = useRef();
  const fileInputRef2 = useRef();

  const handleTextChange = (e) => setText(e.target.value);
  const handleFile1Change = (e) => setFile1(e.target.files[0]);
  const handleFile2Change = (e) => setFile2(e.target.files[0]);

  const handlePredictSingle = async () => {
    const body = [{ Titulo: text, Descripcion: "" }];
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const result = await response.json();
      console.log("Predicción:", result);
    } catch (error) {
      console.error("Error:", error);
    }
  };
  

  const handlePredictMultiple = async () => {
    if (!file1) return;
  
    const formData = new FormData();
    formData.append("file", file1);
  
    try {
      const response = await fetch("http://localhost:8000/predict_csv", {
        method: "POST",
        body: formData
      });
      const result = await response.json();
      console.log("Predicción múltiple:", result);
    } catch (error) {
      console.error("Error:", error);
    }
  };
  

  const handleTrain = async () => {
    if (!file2) return;
  
    const formData = new FormData();
    formData.append("file", file2);
  
    try {
      const response = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData
      });
      const result = await response.json();
      console.log("Entrenamiento:", result);
    } catch (error) {
      console.error("Error:", error);
    }
  };
  

  const renderTabContent = () => {
    let content;
    switch (activeTab) {
      case "unico":
        content = (
          <div className="card fade">
            <input
              type="text"
              placeholder="Inserta tu texto"
              value={text}
              onChange={handleTextChange}
              className="input"
            />
            <button onClick={handlePredictSingle} className="btn">
              PREDECIR
            </button>
          </div>
        );
        break;
      case "multiples":
        content = (
          <div className="card fade">
            <input
              type="file"
              ref={fileInputRef1}
              onChange={handleFile1Change}
              className="file-hidden"
            />
            <button className="file-btn" onClick={() => fileInputRef1.current.click()}>
              <FaFolderOpen style={{ marginRight: "8px" }} />
              {file1 ? file1.name : "Elegir archivo"}
            </button>
            {!file1 && <span className="file-msg">No se ha seleccionado ningún archivo</span>}
            <button onClick={handlePredictMultiple} className="btn">
              PREDECIR
            </button>
          </div>
        );
        break;
      case "entrenar":
        content = (
          <div className="card fade">
            <input
              type="file"
              ref={fileInputRef2}
              onChange={handleFile2Change}
              className="file-hidden"
            />
            <button className="file-btn" onClick={() => fileInputRef2.current.click()}>
              <FaFolderOpen style={{ marginRight: "8px" }} />
              {file2 ? file2.name : "Elegir archivo"}
            </button>
            {!file2 && <span className="file-msg">No se ha seleccionado ningún archivo</span>}
            <button onClick={handleTrain} className="btn">
              ENTRENAR
            </button>
          </div>
        );
        break;
      default:
        content = null;
    }
    return content;
  };

  return (
    <div className="container">
      <div className="tabs">
        <button className={`tab ${activeTab === "unico" ? "active" : ""}`} onClick={() => setActiveTab("unico")}>
          <FaKeyboard /> Único
        </button>
        <button className={`tab ${activeTab === "multiples" ? "active" : ""}`} onClick={() => setActiveTab("multiples")}>
          <FaFileAlt /> Múltiples
        </button>
        <button className={`tab ${activeTab === "entrenar" ? "active" : ""}`} onClick={() => setActiveTab("entrenar")}>
          <FaCogs /> Entrenar
        </button>
      </div>
      {renderTabContent()}
    </div>
  );
}

export default App;
