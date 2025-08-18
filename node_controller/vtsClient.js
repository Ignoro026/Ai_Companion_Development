// vtsClient.js
const WebSocket = require("ws");

let vts;

function connectVTS() {
  vts = new WebSocket("ws://127.0.0.1:8001"); // VTS 기본 포트
  vts.on("open", () => console.log("✅ VTS 연결 성공"));
  vts.on("error", (err) => console.error("❌ VTS 오류:", err.message));
}

function sendExpression(expressionName) {
  if (vts && vts.readyState === WebSocket.OPEN) {
    const payload = {
      apiName: "VTubeStudioPublicAPI",
      requestID: "expressionTrigger",
      messageType: "ExpressionActivationRequest",
      data: { expressionFile: expressionName, active: true },
    };
    vts.send(JSON.stringify(payload));
    console.log("😃 VTS 표정 변경:", expressionName);
  } else {
    console.log("⚠️ VTS 연결이 아직 준비되지 않았습니다.");
  }
}

module.exports = { connectVTS, sendExpression };
