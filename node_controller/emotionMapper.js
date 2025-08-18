// emotionMapper.js
function getMapping(emotion) {
  switch (emotion) {
    case "joy":
      return { obsScene: "HappyScene", vtsExpression: "Smile" };
    case "sadness":
      return { obsScene: "SadScene", vtsExpression: "Sad" };
    default:
      return { obsScene: "DefaultScene", vtsExpression: "Neutral" };
  }
}

module.exports = { getMapping };
