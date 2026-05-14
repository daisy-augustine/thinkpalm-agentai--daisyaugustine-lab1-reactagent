%pip install --quiet streamlit
!pip install ngrok --quiet
!pip install -q groq

!pip install -q streamlit pandas numpy requests plotly scikit-learn groq
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

import os, subprocess, time, re
from google.colab import userdata

os.environ['AVIATIONSTACK_API_KEY'] = userdata.get('AVIATIONSTACK_API_KEY')
os.environ['GROQ_API_KEY']          = userdata.get('GROQ_API_KEY')

!pkill -9 -f streamlit; pkill -9 -f cloudflared
time.sleep(2)

subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.headless', 'true'],
                 env=os.environ.copy(), stdout=open('/content/st.log', 'w'), stderr=subprocess.STDOUT)
time.sleep(8)

cf = subprocess.Popen(['cloudflared', 'tunnel', '--url', 'http://localhost:8501', '--no-autoupdate'],
                      stdout=open('/content/cf.log', 'w'), stderr=subprocess.STDOUT)

for _ in range(30):
    time.sleep(1)
    m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', open('/content/cf.log').read())
    if m: break

print(f"\n  App URL: {m.group(0)}\n  (wait ~20s, keep this cell running)\n")
cf.wait()
