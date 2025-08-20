// Pick API base: local in dev, relative in prod (Vercel rewrite can proxy /api/*)
const API_BASE =
  location.hostname === "127.0.0.1" || location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "";

// Elements
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const themeBtn = document.getElementById("themeBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// Markdown + highlight config
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: (code, lang) => {
    try { return hljs.highlightAuto(code, lang ? [lang] : undefined).value; }
    catch { return code; }
  }
});

// Local storage chat
const LS_KEY = "gpt_chat_history_v1";
function loadHistory(){
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "[]"); }
  catch { return []; }
}
function saveHistory(arr){ localStorage.setItem(LS_KEY, JSON.stringify(arr)); }

let history = loadHistory();
history.forEach(m => addMessage(m.role, m.content, { time: m.time, skipSave:true }));

// Theme
(function initTheme(){
  const saved = localStorage.getItem("theme") || "dark";
  document.body.setAttribute("data-theme", saved);
})();
themeBtn.addEventListener("click", () => {
  const next = document.body.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.body.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

// Health ping (status dot)
async function ping(){
  setStatus("idle");
  try{
    const res = await fetch(`${API_BASE}/`);
    if(res.ok){ setStatus("online"); statusText.textContent = "Online"; }
    else { setStatus("offline"); statusText.textContent = "Offline"; }
  }catch{
    setStatus("offline"); statusText.textContent = "Offline";
  }
}
function setStatus(state){
  statusDot.classList.remove("online","offline","idle");
  statusDot.classList.add(state);
}
ping(); setInterval(ping, 15000);

// Autosize textarea
function autosize(){
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autosize); autosize();

// Enter to send (Shift+Enter = newline)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Buttons
sendBtn.addEventListener("click", sendMessage);
clearBtn.addEventListener("click", () => {
  history = [];
  saveHistory(history);
  chat.innerHTML = "";
  addMessage("assistant", "Chat cleared. How can I help?");
});

// Typing indicator
let typingEl = null;
function showTyping(){
  if (typingEl) return;
  typingEl = messageEl("assistant");
  typingEl.querySelector(".content").innerHTML =
    `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
  chat.appendChild(typingEl);
  scrollToBottom();
}
function hideTyping(){
  if (typingEl){ typingEl.remove(); typingEl = null; }
}

// Create message element
function messageEl(role, timeText){
  const msg = document.createElement("div");
  msg.className = "msg" + (role === "user" ? " right" : "");

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const content = document.createElement("div");
  content.className = "content";

  const meta = document.createElement("div");
  meta.className = "meta";
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = timeText || new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});

  meta.appendChild(time);

  // Copy button for bot messages
  if (role !== "user"){
    const copy = document.createElement("span");
    copy.className = "copy";
    copy.textContent = "Copy";
    copy.title = "Copy response";
    copy.addEventListener("click", () => {
      const text = content.innerText;
      navigator.clipboard.writeText(text);
      copy.textContent = "Copied!";
      setTimeout(()=> copy.textContent = "Copy", 1200);
    });
    meta.appendChild(copy);
  }

  bubble.appendChild(content);
  bubble.appendChild(meta);

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  return msg;
}

function addMessage(role, text, { time=null, skipSave=false } = {}){
  const el = messageEl(role, time);
  // Render markdown for bot; plain for user
  if (role === "assistant"){
    el.querySelector(".content").innerHTML = marked.parse(text || "");
  } else {
    el.querySelector(".content").textContent = text || "";
  }
  chat.appendChild(el);
  scrollToBottom();

  // Highlight code if any
  el.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));

  if (!skipSave){
    history.push({ role, content:text, time: el.querySelector(".time").textContent });
    saveHistory(history);
  }
  return el;
}

function scrollToBottom(){ chat.scrollTop = chat.scrollHeight; }

async function sendMessage(){
  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  input.value = "";
  autosize();

  showTyping();
  try{
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input: text })
    });

    const data = await res.json();
    hideTyping();
    addMessage("assistant", data.response || "(no response)");
  }catch(err){
    hideTyping();
    addMessage("assistant", "⚠️ Network error. Please try again.");
  }
}

// If page is empty (no history), greet the user
if (history.length === 0){
  addMessage("assistant", "Hey! I'm ready. Ask me anything.");
}
