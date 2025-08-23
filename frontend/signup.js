const API_BASE =
  location.hostname === "127.0.0.1" || location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "";

const u = document.getElementById("u");
const p = document.getElementById("p");
const btn = document.getElementById("signupBtn");
const msg = document.getElementById("msg");

btn.addEventListener("click", signup);
p.addEventListener("keydown", (e)=>{ if(e.key==="Enter") signup(); });

async function signup(){
  msg.textContent = "";
  const username = u.value.trim();
  const password = p.value.trim();
  if(!username || !password){ msg.textContent = "Enter username and password."; return; }

  try{
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if(!res.ok){ msg.textContent = data.detail || "Sign up failed"; return; }

    localStorage.setItem("token", data.token);
    localStorage.setItem("username", data.username);
    location.href = "index.html";
  }catch(e){
    msg.textContent = "Network error. Try again.";
  }
}
