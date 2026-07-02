# Deployment Guide

## Option A — ECS / Any Linux VM (simplest)

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd qwen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set the API key (never hardcode it)
export DASHSCOPE_API_KEY=sk-your-key-here

# 4. Run
python app.py
# → Listening on http://0.0.0.0:7860
```

Open port 7860 in your ECS security group inbound rules.

---

## Option B — Docker (ECS + container, or any container platform)

```bash
# Build
docker build -t icap-tutor .

# Run — API key injected at runtime, data directory mounted for persistence
docker run -d \
  -e DASHSCOPE_API_KEY=sk-your-key-here \
  -v $(pwd)/data:/app/data \
  -p 7860:7860 \
  icap-tutor

# Check it started
docker logs <container-id>
# Should see: Running on local URL: http://0.0.0.0:7860
```

---

## Option C — ModelScope Studio (Qwen native)

1. Go to modelscope.cn → Studio → New Space
2. Upload all files (or connect your GitHub repo)
3. Set secret `DASHSCOPE_API_KEY` in Space settings
4. The Space detects `app.py` + `gradio` automatically

---

## Verify the deployment works (do this before recording)

```bash
# 1. Check the app responds
curl -s http://<your-host>:7860 | grep -i "ICAP"

# 2. Check the API key is reaching Qwen — send one message in the UI
#    Expected in decision trace: ① Classify → ② Route → ③ Agent

# 3. Check the audit log is writing
cat data/tool_audit.jsonl   # after a Challenger turn (2nd message on any topic)

# 4. Check sandbox works — paste any Python function and submit
#    Expected: signal = Pass or Fail, not Error
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DASHSCOPE_API_KEY` | Yes | — | Qwen API authentication |
| `ICAP_DATA_DIR` | No | `./data` | Where trajectory + audit logs are written |
| `PORT` | No | `7860` | HTTP port (platforms like FC set this automatically) |

---

## Things that will surprise you on first deploy

**`qwen-coder-plus` availability** — verify this model name works on your endpoint:
```bash
python -c "
from openai import OpenAI
import os
c = OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],
           base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
r = c.chat.completions.create(
    model='qwen-coder-plus',
    messages=[{'role':'user','content':'say ok'}],
    max_tokens=5
)
print('qwen-coder-plus OK:', r.choices[0].message.content)
"
```
If it fails, edit `models.py` and change `QWEN_CODE = "qwen-plus"` as fallback.

**Data directory permissions** — if running as a non-root user in a container:
```bash
# Either mount a writable volume (Option B above) or set ICAP_DATA_DIR to /tmp
export ICAP_DATA_DIR=/tmp/icap-data
```

**rlimit in restricted containers** — the code sandbox applies CPU/memory limits
via `resource.setrlimit`. If the container doesn't allow this (cgroup restrictions),
it degrades gracefully — the wall-clock timeout (5 seconds) still protects you.

**Port binding** — if 7860 is taken, set `PORT=7861` (or whatever is free).
