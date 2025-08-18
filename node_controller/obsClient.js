// obsClient.js
const OBSWebSocket = require("obs-websocket-js");
const obs = new OBSWebSocket();

async function connectOBS() {
  try {
    await obs.connect("ws://127.0.0.1:4455", "비밀번호"); 
    console.log("✅ OBS 연결 성공");
  } catch (err) {
    console.error("❌ OBS 연결 실패:", err.message);
  }
}

async function switchScene(sceneName) {
  try {
    await obs.call("SetCurrentProgramScene", { sceneName });
    console.log("🎬 OBS 장면 전환:", sceneName);
  } catch (err) {
    console.error("❌ 장면 전환 실패:", err.message);
  }
}

module.exports = { connectOBS, switchScene };
