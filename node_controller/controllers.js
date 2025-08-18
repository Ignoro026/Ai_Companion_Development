const axios = require("axios");
const { getMapping } = require("./emotionMapper");
const { connectOBS, switchScene } = require("./obsClient");
const { connectVTS, sendExpression } = require("./vtsClient");

// 실행 시 OBS/VTS 연결
connectOBS();
connectVTS();

const testInput = "hello AI!";

async function main() {
  try {
    // Python AI 서버에 요청
    const res = await axios.post("http://127.0.0.1:5000/chat", {
      text: testInput,
    });

    const { reply, emotion } = res.data;
    console.log("🤖 AI 응답:", reply);
    console.log("😃 감정:", emotion);

    // 감정 → 매핑 가져오기
    const mapping = getMapping(emotion);

    // OBS 장면 전환
    await switchScene(mapping.obsScene);

    // VTS 표정 변경
    sendExpression(mapping.vtsExpression);

  } catch (err) {
    console.error("❌ 오류:", err.message);
  }
}

main();
