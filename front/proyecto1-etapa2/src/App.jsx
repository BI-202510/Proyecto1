import React, { useState, useRef } from "react";
import { FaKeyboard, FaFileAlt, FaCogs, FaFolderOpen } from "react-icons/fa";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [titulo, setTitulo] = useState("");
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [activeTab, setActiveTab] = useState("unico");
  const [errorMessage, setErrorMessage] = useState("");  // Para mostrar errores
  const [predicciones, setPredicciones] = useState(null); // Estado para almacenar resultados de la predicción
  const [prediccionesCSV, setPrediccionesCSV] = useState(null); // Estado para almacenar resultados de predicción de CSV
  const [reentrenamientoResultados, setReentrenamientoResultados] = useState(null); // Estado para resultados del reentrenamiento
  const [metrics, setMetrics] = useState(null); // Estado para las métricas (precision, recall, f1_score)

  const fileInputRef1 = useRef();
  const fileInputRef2 = useRef();

  const handleTextChange = (e) => setText(e.target.value);
  const handleTituloChange = (e) => setTitulo(e.target.value);
  const handleFile1Change = (e) => setFile1(e.target.files[0]);
  const handleFile2Change = (e) => setFile2(e.target.files[0]);

  const clearFile1 = () => setFile1(null);  // Limpiar archivo 1
  const clearFile2 = () => setFile2(null);  // Limpiar archivo 2

  async function handlePredictSingle() {
    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify([
                {
                    Titulo: titulo,
                    Descripcion: text,
                }
            ]),
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        setPredicciones(data); // Guardamos las predicciones en el estado

        // Limpiamos los campos después de predecir
        setTitulo("");
        setText("");
    } catch (error) {
        console.error("Error en la predicción:", error);
        setErrorMessage("Ocurrió un error al hacer la predicción.");
    }
  }

  const handlePredictMultiple = async () => {
    if (!file1) {
      setErrorMessage("Por favor, selecciona un archivo para la predicción.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file1);

    try {
      const response = await fetch("http://localhost:8000/predict_csv", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      setPrediccionesCSV(result); // Guardamos las predicciones del CSV en el estado
      setErrorMessage(""); // Limpiar error si la predicción es exitosa
      clearFile1();  // Limpiar archivo después de la predicción
    } catch (error) {
      console.error("Error:", error);
      setErrorMessage("Ocurrió un error al hacer la predicción.");
    }
  };

  const handleTrain = async () => {
    if (!file2) {
      setErrorMessage("Por favor, selecciona un archivo para el reentrenamiento.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file2);

    try {
      const response = await fetch("http://localhost:8000/retrain", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      setReentrenamientoResultados(result); // Guardamos los resultados del reentrenamiento
      setMetrics(result.metrics); // Guardamos las métricas (precision, recall, f1_score)
      setErrorMessage(""); // Limpiar error si el entrenamiento es exitoso
      clearFile2();  // Limpiar archivo después del reentrenamiento
    } catch (error) {
      console.error("Error:", error);
      setErrorMessage("Ocurrió un error al reentrenar el modelo.");
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
              placeholder="Inserta el título"
              value={titulo || ''}  // Evita que sea undefined
              onChange={handleTituloChange}
              className="input"
            />
            <textarea
              placeholder="Escribe tu texto"
              value={text || ''}  // Evita que sea undefined
              onChange={handleTextChange}
              className="input"
            />
            <button onClick={handlePredictSingle} className="btn">
              PREDECIR
            </button>

            {/* Mostrar el mensaje de error si existe */}
            {errorMessage && <p className="error-message">{errorMessage}</p>}

            {/* Mostrar los resultados de la predicción en una tabla */}
            {predicciones && (
              <div className="resultados">
                <h2>Resultados de la predicción:</h2>
                <table className="resultados-table">
                  <thead>
                    <tr>
                      <th>Predicción</th>
                      <th>Probabilidad</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{predicciones.predictions[0] === 0 ? "Verdadera" : "Falsa"}</td>
                      <td>{predicciones.probabilities[0]}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
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
            {errorMessage && <p className="error-message">{errorMessage}</p>}

            {/* Mostrar los resultados de la predicción múltiple en una tabla */}
            {prediccionesCSV && (
              <div className="resultados">
                <h2>Resultados de las predicciones:</h2>
                <table className="resultados-table">
                  <thead>
                    <tr>
                      <th>Predicción</th>
                      <th>Probabilidad</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediccionesCSV.predictions.map((prediction, index) => (
                      <tr key={index}>
                        <td>{prediction === 0 ? "Verdadera" : "Falsa"}</td>
                        <td>{prediccionesCSV.probabilities[index]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
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
            {errorMessage && <p className="error-message">{errorMessage}</p>}

            {/* Mostrar los resultados del reentrenamiento en una tabla */}
            {reentrenamientoResultados && reentrenamientoResultados.predictions && reentrenamientoResultados.predictions.length > 0 && (
              <div className="resultados">
                <h2>Resultados del reentrenamiento:</h2>
                <table className="resultados-table">
                  <thead>
                    <tr>
                      <th>Predicción</th>
                      <th>Probabilidad</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reentrenamientoResultados.predictions.map((prediction, index) => (
                      <tr key={index}>
                        <td>{prediction === 0 ? "Verdadera" : "Falsa"}</td>
                        <td>{reentrenamientoResultados.probabilities[index]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}


            {/* Mostrar las métricas del reentrenamiento */}
            {metrics && (
              <div className="metrics">
                <h3>Métricas del reentrenamiento:</h3>
                <ul>
                  <li><strong>Precisión:</strong> {metrics.precision}</li>
                  <li><strong>Recall:</strong> {metrics.recall}</li>
                  <li><strong>F1 Score:</strong> {metrics.f1_score}</li>
                </ul>
              </div>
            )}
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
