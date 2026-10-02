import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import Printer3D from "./Printer3D.jsx";
import { askReceipt } from "./api.js";

const GITHUB_URL =
  "https://github.com/pratyushwinorlearn/receipt-vqa";

const HF_URL =
  "https://huggingface.co/spaces/shekharrrr/receiptqa-demo";

const MAX_MB = 8;

const EXAMPLES = [
  "What is the total amount?",
  "How many items are listed?",
  "What is the price of [item]?",
  "How much was paid in cash?",
];

const METRICS = [
  ["84.43%", "Exact match"],
  ["91.07%", "ANLS"],
  ["89.14%", "Numeric accuracy"],
];

const STEPS = [
  ["01", "Receipt image"],
  ["02", "SmolVLM-500M"],
  ["03", "QLoRA adapter"],
  ["04", "Printed answer"],
];

function friendlyError(error) {
  const message = String(
    error?.message || error
  );

  if (
    /quota|exceeded your zerogpu/i.test(
      message
    )
  ) {
    return "The free GPU quota has been reached. Please try again after it resets.";
  }

  if (
    /fetch|network|failed|load failed|connect/i.test(
      message
    )
  ) {
    return "Could not reach the inference server. It may be waking up — wait a moment and try again.";
  }

  if (
    /queue|busy|capacity/i.test(message)
  ) {
    return "The inference server is busy. Try again in a moment.";
  }

  return (
    message ||
    "Something went wrong while running the model."
  );
}

function ExternalIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      className="icon-small"
      aria-hidden="true"
    >
      <path d="M14 5h5v5" />
      <path d="M19 5 11 13" />
      <path d="M19 13v4a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      className="icon-small"
      aria-hidden="true"
    >
      <path d="M5 12h13" />
      <path d="m13 6 6 6-6 6" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className="icon-medium"
      aria-hidden="true"
    >
      <path d="M12 .7a12 12 0 0 0-3.79 23.39c.6.11.82-.26.82-.58v-2.04c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.74.08-.74 1.2.09 1.83 1.23 1.83 1.23 1.07 1.83 2.8 1.3 3.48 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.34-5.47-5.95 0-1.31.47-2.38 1.23-3.22-.12-.3-.53-1.53.12-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 6 0C17.29 5.95 18.3 6.27 18.3 6.27c.65 1.65.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.62-2.81 5.64-5.49 5.94.43.37.81 1.1.81 2.22v3.28c0 .32.22.7.82.58A12 12 0 0 0 12 .7Z" />
    </svg>
  );
}

function HFIcon() {
  return (
    <span
      className="hf-icon"
      aria-hidden="true"
    >
      H
    </span>
  );
}

function ReceiptFlatView({
  preview,
}) {
  if (!preview) {
    return (
      <div className="flat-empty-state">
        <div className="flat-upload-icon">
          <svg
            viewBox="0 0 48 48"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect
              x="12"
              y="7"
              width="24"
              height="34"
              rx="2"
            />
            <path d="M17 14h14" />
            <path d="M17 20h14" />
            <path d="M17 26h9" />
          </svg>
        </div>

        <strong>
          Your receipt will appear here.
        </strong>

        <span>
          Switch back to Printer View after
          uploading a receipt.
        </span>
      </div>
    );
  }

  return (
    <div className="flat-receipt-stage">
      <div className="flat-receipt-top">
        <span>RECEIPT INPUT</span>

      </div>

      <div className="flat-receipt-body">
        <img
          src={preview}
          alt="Uploaded receipt"
        />
      </div>

      <div className="flat-receipt-bottom">
        <span>
          ORIGINAL UPLOAD
        </span>

        <span>
          {preview ? "READY" : ""}
        </span>
      </div>
    </div>
  );
}

function Demo() {
  const [file, setFile] =
    useState(null);

  const [preview, setPreview] =
    useState(null);

  const [question, setQuestion] =
    useState("");

  const [answer, setAnswer] =
    useState("");

  const [asked, setAsked] =
    useState("");

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [dragActive, setDragActive] =
    useState(false);

  const [seconds, setSeconds] =
    useState(0);

  const [viewMode, setViewMode] =
    useState("printer");

  const inputRef =
    useRef(null);

  const questionRef =
    useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }

    const objectUrl =
      URL.createObjectURL(file);

    setPreview(objectUrl);

    return () =>
      URL.revokeObjectURL(
        objectUrl
      );
  }, [file]);

  useEffect(() => {
    if (!loading) {
      setSeconds(0);
      return undefined;
    }

    const interval = setInterval(() => {
      setSeconds(
        (current) => current + 1
      );
    }, 1000);

    return () =>
      clearInterval(interval);
  }, [loading]);

  const acceptFile = useCallback(
    (selectedFile) => {
      setError("");
      setAnswer("");
      setAsked("");

      if (!selectedFile) {
        return;
      }

      if (
        !selectedFile.type.startsWith(
          "image/"
        )
      ) {
        setError(
          "Please upload a PNG, JPG or WebP receipt."
        );
        return;
      }

      if (
        selectedFile.size >
        MAX_MB * 1024 * 1024
      ) {
        setError(
          `Receipt image must be under ${MAX_MB} MB.`
        );
        return;
      }

      setFile(selectedFile);
      setViewMode("printer");
    },
    []
  );

  const reset = () => {
    setFile(null);
    setPreview(null);
    setQuestion("");
    setAnswer("");
    setAsked("");
    setError("");
    setLoading(false);
    setViewMode("printer");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const chooseExample = (
    example
  ) => {
    setQuestion(example);
    setError("");

    requestAnimationFrame(() => {
      const input =
        questionRef.current;

      if (!input) {
        return;
      }

      input.focus();

      const index =
        example.indexOf("[item]");

      if (index >= 0) {
        input.setSelectionRange(
          index,
          index + 6
        );
      }
    });
  };

  const submit = async (
    event
  ) => {
    event.preventDefault();

    if (loading) {
      return;
    }

    if (!file) {
      setError(
        "Upload a receipt before printing."
      );
      return;
    }

    if (!question.trim()) {
      setError(
        "Type a question about the receipt."
      );
      return;
    }

    if (
      /\[item\]/i.test(question)
    ) {
      setError(
        "Replace [item] with a real item name from the receipt."
      );
      return;
    }

    /*
     * Automatically switch back to printer view
     * when the user starts an inference.
     */
    setViewMode("printer");

    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const result =
        await askReceipt(
          file,
          question.trim()
        );

      setAsked(
        question.trim()
      );

      setAnswer(result);
    } catch (submitError) {
      setError(
        friendlyError(
          submitError
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      id="demo"
      className="demo"
    >
      <div className="demo-intro">
        <div>
          <span className="eyebrow">
            LIVE DEMO
          </span>

          <h2>
            Feed the receipt.
            <br />
            Let the machine think.
          </h2>

          <p>
            Upload a receipt, ask a
            question, and watch ReceiptQA
            work like a real thermal printer.
          </p>
        </div>

        <div className="demo-tag">
          THERMAL VQA
        </div>
      </div>

      <div className="demo-layout">
        <div
          className={`three-printer-wrap ${
            dragActive
              ? "drag-active"
              : ""
          } ${
            loading
              ? "stage-printing"
              : ""
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() =>
            setDragActive(false)
          }
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);

            acceptFile(
              event.dataTransfer
                .files?.[0]
            );
          }}
        >
          {viewMode === "printer" ? (
            <Printer3D
              preview={preview}
              printing={loading}
            />
          ) : (
            <ReceiptFlatView
              preview={preview}
            />
          )}

          <div className="demo-controls">
            {!file ? (
              <label
                htmlFor="receipt-upload"
                className="upload-control"
              >
                <input
                  id="receipt-upload"
                  ref={inputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  className="sr-only"
                  onChange={(event) =>
                    acceptFile(
                      event.target
                        .files?.[0]
                    )
                  }
                />

                <span>
                  Upload receipt
                </span>

                <ArrowIcon />
              </label>
            ) : (
              <div className="loaded-control">
                <span>
                  Receipt loaded
                </span>

                <button
                  type="button"
                  onClick={reset}
                >
                  Remove
                </button>
              </div>
            )}

            <div
              className="view-toggle"
              role="tablist"
              aria-label="Receipt display"
            >
              <button
                type="button"
                className={
                  viewMode ===
                  "printer"
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setViewMode(
                    "printer"
                  )
                }
              >
                Printer View
              </button>

              <button
                type="button"
                className={
                  viewMode ===
                  "receipt"
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setViewMode(
                    "receipt"
                  )
                }
              >
                Receipt View
              </button>
            </div>

            {!file && (
              <span className="drop-help">
                Drop a receipt onto the
                printer
              </span>
            )}
          </div>
        </div>

        <div className="query-column">
          <form
            className="query-card"
            onSubmit={submit}
            noValidate
          >
            <div className="query-heading">
              <div>
                <span className="eyebrow">
                  ASK THE MODEL
                </span>

                <h3>
                  Your question
                </h3>
              </div>

              <span className="query-badge">
                VQA
              </span>
            </div>

            <input
              ref={questionRef}
              value={question}
              onChange={(event) =>
                setQuestion(
                  event.target
                    .value
                )
              }
              placeholder="What is the total amount?"
              maxLength={200}
              autoComplete="off"
            />

            <div className="examples">
              {EXAMPLES.map(
                (example) => (
                  <button
                    type="button"
                    key={example}
                    onClick={() =>
                      chooseExample(
                        example
                      )
                    }
                  >
                    {example}
                  </button>
                )
              )}
            </div>

            <button
              className="print-button"
              type="submit"
              disabled={loading}
            >
              <span>
                {loading
                  ? `Printing… ${seconds}s`
                  : "Print the answer"}
              </span>

              <span className="print-button-arrow">
                <ArrowIcon />
              </span>
            </button>

            <p className="query-hint">
              Ask about totals,
              quantities, prices, cash
              payment, item names and
              other receipt fields.
            </p>
          </form>

          <div
            className="result-container"
            aria-live="polite"
          >
            {error && (
              <div
                className="error-box"
                role="alert"
              >
                <strong>
                  Printer stopped
                </strong>

                <span>
                  {error}
                </span>
              </div>
            )}

            {loading &&
              !error && (
                <div className="printing-box">
                  <div className="printing-title">
                    <span className="printing-machine-icon">
                      <i />
                      <i />
                      <i />
                    </span>

                    <div>
                      <strong>
                        Printing your answer
                      </strong>

                      <span>
                        Reading receipt ·{" "}
                        {seconds}s
                      </span>
                    </div>
                  </div>

                  <div className="printing-progress">
                    <span />
                  </div>
                </div>
              )}

            {answer &&
              !loading &&
              !error && (
                <div className="flat-answer-paper receipt-edge">
                  <div className="answer-paper-header">
                    <span>
                      RECEIPTQA
                    </span>

                    <span>
                      RESULT
                    </span>
                  </div>

                  <div className="answer-paper-body">
                    <span className="paper-label">
                      QUESTION
                    </span>

                    <p>
                      {asked}
                    </p>

                    <div className="paper-rule">
                      <span />
                      <span />
                    </div>

                    <span className="paper-label">
                      ANSWER
                    </span>

                    <strong>
                      {answer}
                    </strong>
                  </div>

                  <div className="answer-paper-footer">
                    <span>
                      SMOLVLM-500M
                    </span>

                    <span>
                      QLORA
                    </span>

                    <span>
                      CORD V2
                    </span>
                  </div>
                </div>
              )}

            {!loading &&
              !answer &&
              !error && (
                <div className="empty-answer">
                  <div className="mini-receipt">
                    <span />
                    <span />
                    <span className="wide" />
                    <span />
                    <span className="small" />
                  </div>

                  <div>
                    <strong>
                      Your answer will
                      print here.
                    </strong>

                    <p>
                      Feed a receipt into
                      the printer and ask
                      your first question.
                    </p>
                  </div>
                </div>
              )}
          </div>
        </div>
      </div>

      <p className="privacy">
        Do not upload receipts containing
        sensitive personal information.
        Images are sent to a public demo
        server.
      </p>
    </section>
  );
}

export default function App() {
  return (
    <div className="site">
      <header className="header">
        <div className="content-width">
          <nav className="nav">
            <a
              href="#top"
              className="logo"
            >
              ReceiptQA
            </a>

            <div className="nav-actions">
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noreferrer"
              >
                <GithubIcon />
                GitHub
                <ExternalIcon />
              </a>

              <a
                href={HF_URL}
                target="_blank"
                rel="noreferrer"
              >
                <HFIcon />
                Hugging Face
                <ExternalIcon />
              </a>
            </div>
          </nav>

          <div
            id="top"
            className="hero"
          >
            <div className="hero-pills">
              <span>
                SmolVLM-500M
              </span>

              <span>
                QLoRA r=16
              </span>

              <span>
                CORD v2
              </span>
            </div>

            <h1>
              Ask a receipt.
              <br />
              <em>Get it printed.</em>
            </h1>

            <div className="hero-bottom">
              <p>
                ReceiptQA turns a receipt image
                and a natural-language question
                into a short, exact answer using
                a domain-specialized
                vision-language model.
              </p>

              <a
                href="#demo"
                className="hero-cta"
              >
                Try the printer
                <ArrowIcon />
              </a>
            </div>
          </div>
        </div>
      </header>

      <main>
        <Demo />
        <section className="stats-section">
          <div className="content-width stats-inner">
            <div className="stats-copy">
              <span className="eyebrow">
                FINAL MODEL
              </span>

              <h2>
                Small model.
                <br />
                Focused domain.
              </h2>

              <p>
                SmolVLM-500M-Instruct was
                fine-tuned with QLoRA on CORD
                v2 receipts converted into
                question-answer pairs.
              </p>

              <p>
                Results are from the frozen
                CORD test set: 809 questions
                across 100 receipts.
              </p>
            </div>

            <div className="stats-grid">
              {METRICS.map(
                ([value, label]) => (
                  <div
                    className="stat-card"
                    key={label}
                  >
                    <strong>
                      {value}
                    </strong>

                    <span>
                      {label}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>
        </section>

        <section className="pipeline-section">
          <div className="content-width">
            <span className="eyebrow">
              PIPELINE
            </span>

            <h2>
              From paper to answer.
            </h2>

            <div className="pipeline-grid">
              {STEPS.map(
                ([number, title]) => (
                  <div
                    className="pipeline-item"
                    key={number}
                  >
                    <span>
                      {number}
                    </span>

                    <strong>
                      {title}
                    </strong>

                    {number !== "04" && (
                      <ArrowIcon />
                    )}
                  </div>
                )
              )}
            </div>
          </div>
        </section>

        <section className="about-section">
          <div className="content-width about-inner">
            <div>
              <span className="eyebrow">
                ABOUT RECEIPTQA
              </span>

              <h2>
                How far can a tiny VLM be pushed
                inside one narrow domain?
              </h2>
            </div>

            <div>
              <p>
                ReceiptQA explores
                domain-specific visual question
                answering using a small
                vision-language model, QLoRA
                and a single consumer 8 GB GPU.
              </p>

              <div className="about-actions">
                <a
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noreferrer"
                >
                  <GithubIcon />
                  View source
                  <ExternalIcon />
                </a>

                <a
                  href={HF_URL}
                  target="_blank"
                  rel="noreferrer"
                >
                  <HFIcon />
                  Try on Hugging Face
                  <ExternalIcon />
                </a>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="content-width footer-inner">
          <strong>
            ReceiptQA
          </strong>

          <span>
            SmolVLM-500M · QLoRA · CORD v2
          </span>

          <div
            className="footer-profile"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <img
              src="/profile.jpeg"
              alt="Pratyush"
              style={{
                width: "38px",
                height: "38px",
                borderRadius: "50%",
                objectFit: "cover",
                flexShrink: 0,
                border: "1px solid rgba(255, 255, 255, 0.16)",
              }}
            />

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "4px",
              }}
            >
              <strong
                style={{
                  fontSize: "0.72rem",
                  lineHeight: 1.1,
                }}
              >
                Shekhar Pratyush
              </strong>

              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.48rem",
                }}
              >
                BTech CSE · AI/ML
              </span>

              <div
                style={{
                  display: "flex",
                  gap: "10px",
                  marginTop: "1px",
                }}
              >
                <a
                  href="https://www.linkedin.com/in/shekhar-pratyush-445362327"
                  target="_blank"
                  rel="noreferrer"
                >
                  LinkedIn
                </a>

                <a href="mailto:pratyushqgis22@gmail.com">
                  Email
                </a>
              </div>
            </div>
          </div>

          <div className="footer-links">
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>

            <a
              href={HF_URL}
              target="_blank"
              rel="noreferrer"
            >
              Hugging Face
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}