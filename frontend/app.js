
(() => {
  const $ = id => document.getElementById(id);

  const video = $("video");
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");

  const badge = $("badge");
  const current = $("current");
  const barFill = $("barFill");
  const statusText = $("statusText");
  const confText = $("confText");
  const sentenceEl = $("sentence");
  const errorEl = $("error");
  const placeholder = $("cameraPlaceholder");

  const startBtn = $("startBtn");
  const stopBtn = $("stopBtn");
  const clearBtn = $("clearBtn");
  const speakBtn = $("speakBtn");
  const pauseBtn = $("pauseBtn");
  const resumeBtn = $("resumeBtn");
  const repeatBtn = $("repeatBtn");
  const autoSpeak = $("autoSpeak");

  const FRAME_INTERVAL_MS = 1000 / 15;

  let ws = null;
  let stream = null;
  let running = false;
  let lastSent = 0;
  let sentence = "";
  let history = [];
  let lastSpoken = "";

  function setError(message) {
    errorEl.textContent = message || "";
  }

  // Dark and light theme
  $("themeBtn").onclick = () => {
    document.body.classList.toggle("light-theme");

    const light = document.body.classList.contains("light-theme");
    $("themeBtn").textContent = light ? "🌙" : "☀️";

    localStorage.setItem("signbridge-theme", light ? "light" : "dark");
  };

  if (localStorage.getItem("signbridge-theme") === "light") {
    document.body.classList.add("light-theme");
    $("themeBtn").textContent = "🌙";
  }

  // Speech
  function speak(text) {
    if (!text || !("speechSynthesis" in window)) {
      setError("Speech is not supported in this browser.");
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;

    window.speechSynthesis.speak(utterance);
    lastSpoken = text;
  }

  speakBtn.onclick = () => speak(sentence);

  pauseBtn.onclick = () => {
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.pause();
    }
  };

  resumeBtn.onclick = () => {
    window.speechSynthesis.resume();
  };

  repeatBtn.onclick = () => speak(lastSpoken || sentence);

  // Translation history
  function updateHistory(text) {
    if (!text) return;

    if (history[0] === text) return;

    history.unshift(text);
    history = history.slice(0, 5);

    const list = $("historyList");
    list.innerHTML = "";

    history.forEach(item => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
  }

  function stopCameraTracks() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    video.srcObject = null;
    placeholder.style.display = "flex";
  }

  // Start camera
  async function start() {
    setError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera access is unavailable. Open the app using localhost or HTTPS.");
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      video.srcObject = stream;
      await video.play();

      placeholder.style.display = "none";

      canvas.width = 480;
      canvas.height = 360;

      const protocol = location.protocol === "https:" ? "wss" : "ws";

      ws = new WebSocket(`${protocol}://${location.host}/ws`);

      ws.onopen = () => {
        running = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        clearBtn.disabled = false;

        badge.textContent = "Connected";

        sendFrame();
      };

      ws.onmessage = event => {
        try {
          handle(JSON.parse(event.data));
        } catch {
          setError("Invalid response received from server.");
          stop();
        }
      };

      ws.onerror = () => {
        setError("WebSocket error. Check whether the server is running.");
      };

      ws.onclose = () => {
        if (running) stop();
      };

    } catch (error) {
      stopCameraTracks();
      setError("Camera error: " + error.message);
    }
  }

  // Stop camera
  function stop() {
    running = false;

    const socket = ws;
    ws = null;

    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close();
    }

    stopCameraTracks();
    window.speechSynthesis?.cancel();

    startBtn.disabled = false;
    stopBtn.disabled = true;
    clearBtn.disabled = true;

    badge.textContent = "Camera off";
    current.textContent = "Waiting...";
    statusText.textContent = "Idle";
    confText.textContent = "";
    barFill.style.width = "0%";
  }

  // Send frames to FastAPI
  function sendFrame() {
    if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;

    const wait = Math.max(
      0,
      FRAME_INTERVAL_MS - (performance.now() - lastSent)
    );

    setTimeout(() => {
      if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;

      if (video.readyState < 2) {
        sendFrame();
        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(blob => {
        if (!blob || !running || !ws || ws.readyState !== WebSocket.OPEN) return;

        lastSent = performance.now();
        ws.send(blob);
      }, "image/jpeg", 0.7);

    }, wait);
  }

  // Handle backend recognition
  function handle(message) {
    if (message.status === "no_model") {
      setError(message.message || "No trained recognition model found.");
      stop();
      return;
    }

    if (typeof message.sentence === "string") {
      sentence = message.sentence;
      sentenceEl.textContent = sentence || "Your translated words will appear here...";
      if (sentence) updateHistory(sentence);
    }

    switch (message.status) {
      case "idle":
        statusText.textContent = "Show your hands to the camera";
        current.textContent = "Waiting...";
        confText.textContent = "";
        barFill.style.width = "0%";
        badge.textContent = "Waiting for hands";
        break;

      case "collecting":
        statusText.textContent = "Collecting gesture frames...";
        barFill.style.width = Math.round(message.progress * 100) + "%";
        badge.textContent = "Reading sign";
        break;

      case "predicting":
        statusText.textContent = "Recognizing gesture";

        current.textContent = (message.label || "Unknown").replace(/_/g, " ");

        const confidence = Math.round((message.confidence || 0) * 100);

        confText.textContent = confidence + "% confidence";
        barFill.style.width = confidence + "%";

        badge.textContent = "AI Recognizing";

        if (message.new && message.word && autoSpeak.checked) {
          speak(message.word);
        }
        break;

      case "cleared":
        sentence = "";
        sentenceEl.textContent = "Your translated words will appear here...";
        return;

      case "bad_frame":
        statusText.textContent = "Unable to read camera frame";
        break;
    }

    sendFrame();
  }

  // Clear translated text
  clearBtn.onclick = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send("clear");
    }
  };

  startBtn.onclick = start;
  stopBtn.onclick = stop;

  window.addEventListener("beforeunload", () => {
    stopCameraTracks();
    window.speechSynthesis?.cancel();
  });

})();
