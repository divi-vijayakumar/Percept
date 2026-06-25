const canvas = document.querySelector("#scene");
const ctx = canvas.getContext("2d");
const incident = document.querySelector("#incident");
const frameLabel = document.querySelector("#frameLabel");
const frameState = document.querySelector("#frameState");

const frames = [
  { t: 0, person: false, state: "quiet", label: "empty room" },
  { t: 1, person: true, state: "edge escalates", label: "person enters zone" },
  { t: 2, person: true, state: "refractory suppresses repeat", label: "same person remains" },
  { t: 3, person: false, state: "confident false rearms", label: "zone clear" }
];

let i = 0;

function draw(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#253244";
  ctx.fillRect(74, 72, 572, 272);
  ctx.strokeStyle = frame.person ? "#21a67a" : "#8aa0b8";
  ctx.lineWidth = 5;
  ctx.strokeRect(250, 98, 230, 210);
  ctx.fillStyle = "#d8dee9";
  ctx.font = "700 20px system-ui";
  ctx.fillText("restricted zone", 84, 54);
  ctx.fillStyle = "#90a4bb";
  ctx.font = "16px system-ui";
  ctx.fillText(`t=${frame.t}s · ${frame.state}`, 84, 374);

  if (frame.person) {
    ctx.fillStyle = "#f3c46b";
    ctx.beginPath();
    ctx.arc(365, 164, 32, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e5edf7";
    ctx.fillRect(328, 202, 74, 94);
  }

  frameLabel.textContent = frame.label;
  frameState.textContent = frame.state;
  if (frame.t === 1) {
    incident.className = "incident fire";
    incident.innerHTML = "<b>FIRE zone</b><span>Alert: person in zone</span><small>snippet: 2-frame pre-roll · webhook-ready</small>";
  } else if (frame.t === 3) {
    incident.className = "incident idle";
    incident.textContent = "Re-armed after confident false";
  }
}

draw(frames[0]);
setInterval(() => {
  i = (i + 1) % frames.length;
  draw(frames[i]);
}, 1400);
