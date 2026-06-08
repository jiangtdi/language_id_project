import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_LANGUAGES = ["Chinese", "English", "French", "Japanese", "Korean"];
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

export default function App() {
  const [modelInfo, setModelInfo] = useState({
    model_name: "OWSM-CTC v4 + Conv-Adapter PEFT",
    supported_languages: DEFAULT_LANGUAGES,
    device: "loading",
  });
  const [status, setStatus] = useState("正在连接识别服务...");
  const [result, setResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [error, setError] = useState("");

  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    let ignore = false;
    async function loadModelInfo() {
      try {
        const response = await fetch(`${API_BASE}/api/model-info`);
        if (!response.ok) {
          throw new Error("模型服务未就绪");
        }
        const data = await response.json();
        if (!ignore) {
          setModelInfo(data);
          setStatus("系统已连接，可以上传音频或现场录音。");
        }
      } catch (loadError) {
        if (!ignore) {
          setStatus("识别服务连接失败，请确认后端已启动。");
          setError(loadError.message);
        }
      }
    }
    loadModelInfo();
    return () => {
      ignore = true;
      clearInterval(timerRef.current);
    };
  }, []);

  const topPrediction = result?.probabilities?.[0];
  const sortedProbabilities = useMemo(() => result?.probabilities ?? [], [result]);

  const predictAudio = useCallback(async (file) => {
    setIsPredicting(true);
    setError("");
    setResult(null);
    setStatus("正在分析音频，请稍候...");

    try {
      const formData = new FormData();
      formData.append("file", file, file.name || "recording.wav");
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "识别失败");
      }
      setResult(data);
      setStatus("识别完成");
    } catch (predictError) {
      setError(predictError.message);
      setStatus("识别失败，请检查音频格式或重新录制一段清晰人声。");
    } finally {
      setIsPredicting(false);
    }
  }, []);

  const onFileChange = useCallback(
    (event) => {
      const file = event.target.files?.[0];
      if (file) {
        predictAudio(file);
      }
      event.target.value = "";
    },
    [predictAudio]
  );

  const startRecording = useCallback(async () => {
    setError("");
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      chunksRef.current = [];

      processor.onaudioprocess = (event) => {
        chunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
      streamRef.current = stream;
      audioContextRef.current = audioContext;
      processorRef.current = processor;
      setRecordSeconds(0);
      setIsRecording(true);
      setStatus("正在录音，建议录制 5~15 秒清晰人声。");
      timerRef.current = setInterval(() => {
        setRecordSeconds((seconds) => seconds + 1);
      }, 1000);
    } catch (recordError) {
      setError(recordError.message);
      setStatus("无法访问麦克风，请检查浏览器权限。");
    }
  }, []);

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current);
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    audioContextRef.current?.close();

    const wavBlob = encodeWav(chunksRef.current, 16000);
    setIsRecording(false);
    predictAudio(new File([wavBlob], "recording.wav", { type: "audio/wav" }));
  }, [predictAudio]);

  return (
    <main className="app-shell">
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />

      <section className="hero-grid">
        <article className="hero-panel glass-panel">
          <div className="eyebrow">
            <span className="pulse-dot" />
            语音语种识别演示
          </div>
          <h1>多语种语音语种识别系统</h1>
          <p className="hero-copy">
            上传一段人声，或直接用麦克风录音，系统会判断这段语音更接近哪一种语言，
            并给出每个语种的置信度。
          </p>

          <div className="language-cloud" aria-label="支持语种">
            {modelInfo.supported_languages.map((language) => (
              <span key={language}>{language}</span>
            ))}
          </div>

          <div className="action-row">
            <label className={classNames("primary-action", isPredicting && "disabled")}>
              <input type="file" accept="audio/*" onChange={onFileChange} disabled={isPredicting} />
              上传音频识别
            </label>
            <button
              type="button"
              className={classNames("record-action", isRecording && "recording")}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isPredicting}
            >
              {isRecording ? `停止录音 ${recordSeconds}s` : "现场录音"}
            </button>
          </div>

          <div className="status-line" role="status">
            <span className={classNames("status-orb", isPredicting && "spinning", isRecording && "recording")} />
            {status}
          </div>

          {error && <div className="error-card">错误：{error}</div>}
        </article>

        <aside className="model-card glass-panel">
          <div className="model-card-top">
            <span>使用流程</span>
            <strong>{modelInfo.device === "loading" ? "准备中" : "已连接"}</strong>
          </div>
          <h2>从语音到语种</h2>
          <p>演示时只需要准备一段人声样本，系统会完成音频分析，并把判断结果以概率形式展示出来。</p>
          <div className="compact-steps">
            {["上传音频或现场录音", "分析语音中的声学特征", "判断最可能的语种", "展示各语种置信度"].map(
              (item, index) => (
                <div className="pipeline-step" key={item}>
                  <b>{index + 1}</b>
                  <span>{item}</span>
                </div>
              )
            )}
          </div>
        </aside>
      </section>

      <section className="result-grid">
        <article className="result-panel glass-panel">
          <div className="section-heading">
            <span>识别结果</span>
            <h2>识别结果</h2>
          </div>

          {!result && !isPredicting && (
            <div className="empty-state">
              <div className="wave-mark">
                <span />
                <span />
                <span />
                <span />
              </div>
              <p>上传音频或点击录音后，这里会显示预测语种和五类概率。</p>
            </div>
          )}

          {isPredicting && (
            <div className="skeleton-area">
              <div className="skeleton skeleton-title" />
              <div className="skeleton skeleton-bar" />
              <div className="skeleton skeleton-bar short" />
            </div>
          )}

          {result && (
            <>
              <div className="winner-card">
                <div>
                  <span>预测语种</span>
                  <strong>{result.predicted_language}</strong>
                </div>
                <div className="confidence-ring">
                  <b>{formatPercent(result.confidence)}</b>
                  <span>confidence</span>
                </div>
              </div>

              <div className="probability-list">
                {sortedProbabilities.map((item, index) => (
                  <div className="probability-row" key={item.label}>
                    <div className="rank">{String(index + 1).padStart(2, "0")}</div>
                    <div className="probability-main">
                      <div className="probability-label">
                        <span>{item.label}</span>
                        <strong>{formatPercent(item.probability)}</strong>
                      </div>
                      <div className="probability-track">
                        <span style={{ width: formatPercent(item.probability) }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </article>
      </section>
    </main>
  );
}

function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const samples = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    samples.set(chunk, offset);
    offset += chunk.length;
  });

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPcm(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function floatTo16BitPcm(view, offset, input) {
  for (let index = 0; index < input.length; index += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}
