import { useState } from "react";
import axios from "axios";

function App() {
  const [pdf, setPdf] = useState(null);
  const [message, setMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);

  const uploadPDF = async () => {
    const formData = new FormData();
    formData.append("pdf", pdf);

    const res = await axios.post("http://127.0.0.1:5000/upload", formData);

    setMessage(res.data.message + " | Chunks: " + res.data.total_chunks);
  };

  const askQuestion = async () => {
    const res = await axios.post("http://127.0.0.1:5000/ask", {
      question: question,
    });

    setAnswer(res.data.answer);
    setSources(res.data.sources);
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>StudyMate AI</h1>
      <h3>RAG-based PDF Question Answering Assistant</h3>

      <div>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setPdf(e.target.files[0])}
        />
        <button onClick={uploadPDF}>Upload PDF</button>
      </div>

      <p>{message}</p>

      <hr />

      <div>
        <input
          type="text"
          placeholder="Ask a question from PDF"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ width: "400px", padding: "10px" }}
        />
        <button onClick={askQuestion}>Ask</button>
      </div>

      <h2>Answer</h2>
      <p>{answer}</p>

      <h2>Retrieved Sources</h2>
      {sources.map((src, index) => (
        <div key={index} style={{ border: "1px solid gray", margin: "10px", padding: "10px" }}>
          <p><b>Page:</b> {src.page}</p>
          <p><b>Similarity Score:</b> {src.score.toFixed(4)}</p>
          <p>{src.text}</p>
        </div>
      ))}
    </div>
  );
}

export default App;