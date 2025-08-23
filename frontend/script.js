/* ===== ENV / API BASE (dev vs prod) ===== */
const API_BASE =
  location.hostname === "127.0.0.1" || location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : ""; // on Vercel use /api/* rewrite

/* ===== AUTH HELPERS ===== */
function token() { return localStorage.getItem("token") || ""; }
function authHeaders() { const t = token(); return t ? { Authorization: `Bearer ${t}` } : {}; }
function ensureLoggedIn(){ if(!token()) location.href = "login.html"; }
ensureLoggedIn();

/* ===== ELEMENTS ===== */
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const logoutBtn = document.getElementById("logoutBtn");
const themeBtn = document.getElementById("themeBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const sidebar = document.getElementById("sidebar");
const chatList = document.getElementById("chatList");
const newChatBtn = document.getElementById("newChatBtn");
const toggleSidebarBtn = document.getElementById("toggleSidebar");

/* NEW: image UI elements */
const uploadBtn = document.getElementById("uploadBtn");
const genBtn = document.getElementById("genBtn");
const imgInput = document.getElementById("imgInput");

/* ===== MARKDOWN / HIGHLIGHT ===== */
if (window.marked) {
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: (code, lang) => {
      try { return hljs.highlightAuto(code, lang ? [lang] : undefined).value; }
      catch { return code; }
    }
  });
}

/* ===== STATE (server-backed) ===== */
let chats = [];    // [{id,title,updated_at}]
let activeId = ""; // current chat id

/* ===== API WRAPPER ===== */
async function api(path, options={}){
  const opts = {
    ...options,
    headers: {
      ...(options.headers || {}),
      "Content-Type":"application/json",
      ...authHeaders(),
    }
  };
  // Don't set content-type for FormData calls
  if (options.body instanceof FormData) {
    delete opts.headers["Content-Type"];
  }
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (res && res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    location.href = "login.html";
    return;
  }
  return res.json();
}

/* ===== THEME ===== */
(function initTheme(){
  const saved = localStorage.getItem("theme") || "dark";
  document.body.setAttribute("data-theme", saved);
})();
themeBtn.addEventListener("click", () => {
  const next = document.body.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.body.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

/* ===== LOGOUT ===== */
logoutBtn.addEventListener("click", ()=>{
  localStorage.removeItem("token");
  localStorage.removeItem("username");
  location.href = "login.html";
});

/* ===== HEALTH PING ===== */
async function ping(){
  setStatus("idle");
  try{
    const res = await fetch(`${API_BASE}/`);
    if(res.ok){ setStatus("online"); statusText.textContent = "Online"; }
    else { setStatus("offline"); statusText.textContent = "Offline"; }
  }catch{ setStatus("offline"); statusText.textContent = "Offline"; }
}
function setStatus(state){
  statusDot.classList.remove("online","offline","idle");
  statusDot.classList.add(state);
}
ping(); setInterval(ping, 15000);

/* ===== TEXTAREA ===== */
function autosize(){
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autosize); autosize();
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* ===== SIDEBAR ===== */
toggleSidebarBtn?.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});
newChatBtn.addEventListener("click", () => newChat());

function renderSidebar(){
  chatList.innerHTML = "";
  if (!chats || chats.length === 0) return;
  chats.forEach(c => {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === activeId ? " active" : "");
    item.innerHTML = `
      <div class="logo">💬</div>
      <div class="title">${(c.title || "New chat")}</div>
      <div class="time">${c.updated_at ? new Date(c.updated_at*1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}) : ""}</div>
      <div class="trash" title="Delete">🗑</div>
    `;
    item.addEventListener("click", (e)=>{
      if (e.target && e.target.classList.contains("trash")) return;
      switchChat(c.id);
    });
    item.querySelector(".trash").addEventListener("click", async (e)=>{
      e.stopPropagation();
      await api(`/api/chats/${c.id}`, { method: "DELETE" });
      await loadChats();
    });
    chatList.appendChild(item);
  });
}

/* ===== MESSAGE UI ===== */
let typingEl = null;
function messageEl(role, timeText){
  const msg = document.createElement("div");
  msg.className = "msg" + (role === "user" ? " right" : "");
  const avatar = document.createElement("div"); avatar.className = "avatar"; avatar.textContent = role === "user" ? "🧑" : "🤖";
  const bubble = document.createElement("div"); bubble.className = "bubble";
  const content = document.createElement("div"); content.className = "content";
  const meta = document.createElement("div"); meta.className = "meta";
  const time = document.createElement("span"); time.className = "time"; time.textContent = timeText || new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
  meta.appendChild(time);
  if (role !== "user"){
    const copy = document.createElement("span"); copy.className = "copy"; copy.title = "Copy response"; copy.textContent = "Copy";
    copy.addEventListener("click", ()=>{ navigator.clipboard.writeText(content.innerText); copy.textContent="Copied!"; setTimeout(()=>copy.textContent="Copy", 1100); });
    meta.appendChild(copy);
  }
  bubble.appendChild(content); bubble.appendChild(meta);
  msg.appendChild(avatar); msg.appendChild(bubble);
  return { msg, content };
}

/* Render markdown for BOTH user & assistant so uploaded/generated images appear */
function addMessageEl(role, text){
  const { msg, content } = messageEl(role);
  if (window.marked) content.innerHTML = marked.parse(text || "");
  else content.textContent = text || "";
  chat.appendChild(msg);
  chat.querySelectorAll("pre code").forEach(block => window.hljs && hljs.highlightElement(block));
  chat.scrollTop = chat.scrollHeight;
}
function showTyping(){
  if (typingEl) return;
  const { msg, content } = messageEl("assistant");
  content.innerHTML = `<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>`;
  typingEl = msg; chat.appendChild(typingEl); chat.scrollTop = chat.scrollHeight;
}
function hideTyping(){ if (typingEl){ typingEl.remove(); typingEl = null; } }

/* ===== LOAD & RENDER CHAT ===== */
async function loadChats(){
  const data = await api("/api/chats");
  chats = (data && data.chats) ? data.chats : [];
  if(!activeId && chats.length) activeId = chats[0].id;
  renderSidebar();
  await renderChat();
}

async function renderChat(){
  chat.innerHTML = "";
  if(!activeId){
    addMessageEl("assistant", "Hey! I'm ready. Create a new chat or select one from the left.");
    return;
  }
  const data = await api(`/api/chats/${activeId}`);
  const msgs = (data && data.messages) ? data.messages : [];
  if(msgs.length === 0){
    addMessageEl("assistant", "This chat is empty. Say something!");
  } else {
    for(const m of msgs){
      addMessageEl(m.role === "assistant" ? "assistant" : "user", m.content);
    }
  }
}

async function newChat(){
  const res = await api("/api/chats/new", { method: "POST" });
  activeId = res.chat_id;
  // Also reset server memory for this chat
  await api("/api/reset", {
    method: "POST",
    body: JSON.stringify({ chat_id: activeId })
  });
  await loadChats();
  input.focus();
}

async function switchChat(id){
  activeId = id;
  renderSidebar();
  await renderChat();
  sidebar.classList.remove("open");
}

/* ===== SEND FLOW ===== */
sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage(){
  const text = input.value.trim();
  if(!text) return;
  if(!activeId){
    const res = await api("/api/chats/new", { method: "POST" });
    activeId = res.chat_id;
  }

  addMessageEl("user", text);
  input.value = ""; autosize();

  showTyping();
  try{
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ user_input: text, chat_id: activeId })
    });
    hideTyping();
    addMessageEl("assistant", (data && data.response) || "(no response)");
    await loadChats(); // refresh sidebar times/titles
  }catch(e){
    hideTyping();
    addMessageEl("assistant", "⚠️ Network error. Please try again.");
  }
}

/* Clear current conversation (server + memory) */
clearBtn.addEventListener("click", async () => {
  if(!activeId) return;
  await api(`/api/chats/${activeId}/clear`, { method: "POST" });
  await api("/api/reset", {
    method: "POST",
    body: JSON.stringify({ chat_id: activeId })
  });
  await renderChat();
});

/* ===== NEW: IMAGE FEATURES ===== */

/* Analyze an uploaded image with Gemini Vision */
uploadBtn.addEventListener("click", () => imgInput.click());

imgInput.addEventListener("change", async () => {
  const file = imgInput.files && imgInput.files[0];
  if (!file) return;

  const prompt = window.prompt(
    "Describe what you want me to do with this image:",
    "Describe this image"
  );

  // show the uploaded image as a user message
  const url = URL.createObjectURL(file);
  addMessageEl("user", `(uploaded image)\n\n![image](${url})`);
  showTyping();

  try {
    const form = new FormData();
    form.append("image", file);
    form.append("prompt", prompt || "Describe this image");
    form.append("chat_id", activeId || "default");

    const res = await fetch(`${API_BASE}/api/vision`, {
      method: "POST",
      headers: { ...authHeaders() }, // IMPORTANT: don't set content-type when sending FormData
      body: form,
    });

    const data = await res.json();
    hideTyping();
    addMessageEl("assistant", data.response || "(no response)");
    await loadChats();
  } catch (e) {
    hideTyping();
    addMessageEl("assistant", "⚠️ Image analyze failed.");
  } finally {
    imgInput.value = "";
  }
});

/* Generate an image with Imagen */
genBtn.addEventListener("click", async () => {
  const prompt = window.prompt(
    "Image prompt (what to create)?",
    "A cute corgi wearing sunglasses, studio lighting"
  );
  if (!prompt) return;

  addMessageEl("user", `Generate: ${prompt}`);
  showTyping();

  try {
    const res = await fetch(`${API_BASE}/api/image/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ prompt, aspect_ratio: "1:1", count: 1 }),
    });

    const data = await res.json();
    hideTyping();

    if (data.images && data.images.length) {
      data.images.forEach((src) => addMessageEl("assistant", `![generated](${src})`));
      await loadChats();
    } else {
      addMessageEl("assistant", "No image returned.");
    }
  } catch (e) {
    hideTyping();
    addMessageEl("assistant", "⚠️ Image generation failed.");
  }
});

/* ===== INIT ===== */
(async function init(){
  await loadChats();
})();
