import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLM:
    def __init__(self, model_name="skt/kogpt2-base-v2", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 LLM 로딩: {self.model_name} (device={self.device})")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✅ LLM 준비 완료")

    @torch.inference_mode()
    def generate_reply(self, user_text: str, lang_hint="ko"):
        # 한국어 기준 간단 프롬프트
        prompt = f"사용자: {user_text}\nAI:"
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=self.tok.eos_token_id
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        if "AI:" in text:
            text = text.split("AI:", 1)[-1].strip()
        return text
