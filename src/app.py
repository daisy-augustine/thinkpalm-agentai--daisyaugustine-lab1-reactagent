%%writefile app.py

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import os
import json
from groq import Groq

AVIATIONSTACK_ACCESS_KEY = os.getenv('AVIATIONSTACK_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"
AIRPORTS_API_URL = "https://api.travelpayouts.com/data/en/airports.json"

ADSB_SOURCES = [
    ('adsb.lol',       'https://api.adsb.lol/v2'),
    ('adsb.fi',        'https://opendata.adsb.fi/api/v2'),
    ('airplanes.live', 'https://api.airplanes.live/v2'),
]

# =====================================================================
# HELPERS
# =====================================================================

@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_airports():
    try:
        r = requests.get(AIRPORTS_API_URL, timeout=30); r.raise_for_status()
        return {a['code']: a for a in r.json()
                if a.get('code') and len(a['code']) == 3 and a.get('coordinates', {}).get('lat')}
    except requests.exceptions.RequestException:
        return {}

def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict): return default
        d = d.get(k)
        if d is None: return default
    return d

def _fmt_time(s):
    if not s: return None
    ts = pd.to_datetime(s, errors='coerce')
    return None if pd.isna(ts) else ts.strftime('%I:%M %p').lstrip('0')

def _status_color(status):
    return {
        'scheduled': ('#3B82F6', '#DBEAFE'),
        'active':    ('#10B981', '#D1FAE5'),
        'landed':    ('#6B7280', '#F3F4F6'),
        'cancelled': ('#EF4444', '#FEE2E2'),
        'incident':  ('#F59E0B', '#FEF3C7'),
        'diverted':  ('#F59E0B', '#FEF3C7'),
    }.get((status or '').lower(), ('#6366F1', '#E0E7FF'))

# =====================================================================
# TOOLS
# =====================================================================

def tool_get_flight_details(flight_number: str) -> dict:
    if not AVIATIONSTACK_ACCESS_KEY:
        return {'error': 'AviationStack API key not set'}
    try:
        r = requests.get(f"{AVIATIONSTACK_BASE_URL}/flights",
                         params={'access_key': AVIATIONSTACK_ACCESS_KEY,
                                 'flight_iata': flight_number.upper(), 'limit': 1},
                         timeout=30)
        if r.status_code in (401, 403):
            try:
                err = r.json().get('error', {})
                return {'error': f"{err.get('code')}: {err.get('info')}"}
            except Exception:
                return {'error': f'HTTP {r.status_code}'}
        r.raise_for_status()
        payload = r.json()
        if payload.get('error'):
            return {'error': payload['error'].get('info') or str(payload['error'])}
        flights = payload.get('data') or []
        if not flights:
            return {'error': f'No flight found for {flight_number}'}
        f = flights[0]
        return {
            'flight_iata':     _g(f, 'flight', 'iata'),
            'flight_icao':     _g(f, 'flight', 'icao'),
            'flight_number':   _g(f, 'flight', 'number'),
            'airline':         _g(f, 'airline', 'name'),
            'airline_icao':    _g(f, 'airline', 'icao'),
            'airline_iata':    _g(f, 'airline', 'iata'),
            'status':          f.get('flight_status'),
            'dep_iata':        _g(f, 'departure', 'iata'),
            'dep_airport':     _g(f, 'departure', 'airport'),
            'dep_terminal':    _g(f, 'departure', 'terminal'),
            'dep_gate':        _g(f, 'departure', 'gate'),
            'dep_scheduled':   _g(f, 'departure', 'scheduled'),
            'dep_actual':      _g(f, 'departure', 'actual'),
            'arr_iata':        _g(f, 'arrival', 'iata'),
            'arr_airport':     _g(f, 'arrival', 'airport'),
            'arr_terminal':    _g(f, 'arrival', 'terminal'),
            'arr_gate':        _g(f, 'arrival', 'gate'),
            'arr_scheduled':   _g(f, 'arrival', 'scheduled'),
            'arr_estimated':   _g(f, 'arrival', 'estimated'),
            'arr_actual':      _g(f, 'arrival', 'actual'),
            'aircraft_reg':    _g(f, 'aircraft', 'registration'),
            'aircraft_type':   _g(f, 'aircraft', 'iata'),
            'aircraft_icao24': _g(f, 'aircraft', 'icao24'),
            'live_latitude':   _g(f, 'live', 'latitude'),
            'live_longitude':  _g(f, 'live', 'longitude'),
            'live_altitude':   _g(f, 'live', 'altitude'),
            'live_speed':      _g(f, 'live', 'speed_horizontal'),
        }
    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {e}'}

def tool_get_live_position(callsign: str = "", registration: str = "", icao24: str = "",
                           airline_icao: str = "", flight_number: str = "",
                           flight_iata: str = "") -> dict:
    headers = {'User-Agent': 'flight-agent/1.0'}
    queries = []

    if icao24:
        hx = icao24.lower().strip()
        if hx and hx != 'none': queries.append(('icao', hx))

    if registration:
        reg = registration.upper().strip()
        if reg and reg != 'NONE':
            queries.append(('registration', reg))
            if '-' in reg: queries.append(('registration', reg.replace('-', '')))

    callsigns = set()
    def _add_cs(c):
        if c:
            c = str(c).upper().strip().replace(' ', '').replace('-', '')
            if c and c != 'NONE' and len(c) >= 3: callsigns.add(c)

    _add_cs(callsign)
    if airline_icao and flight_number:
        _add_cs(f"{airline_icao}{flight_number}")
        if str(flight_number).isdigit():
            _add_cs(f"{airline_icao}{int(flight_number):d}")
            _add_cs(f"{airline_icao}{int(flight_number):03d}")
            _add_cs(f"{airline_icao}{int(flight_number):04d}")
    _add_cs(flight_iata)
    if airline_icao and flight_iata:
        num_part = ''.join(c for c in flight_iata if c.isdigit())
        if num_part:
            _add_cs(f"{airline_icao}{num_part}")

    for cs in callsigns:
        queries.append(('callsign', cs))

    if not queries:
        return {'error': 'No identifiers provided', 'tried': []}

    tried = []
    for lookup_type, value in queries:
        for source, base in ADSB_SOURCES:
            tried.append(f"{source}:{lookup_type}={value}")
            try:
                r = requests.get(f"{base}/{lookup_type}/{value}", headers=headers, timeout=8)
                r.raise_for_status()
                ac = (r.json() or {}).get('ac') or []
                for a in ac:
                    if a.get('lat') is not None and a.get('lon') is not None:
                        return {
                            'source':       source,
                            'lookup_by':    f"{lookup_type}={value}",
                            'callsign':     (a.get('flight') or '').strip(),
                            'icao24':       (a.get('hex') or '').upper(),
                            'latitude':     float(a['lat']),
                            'longitude':    float(a['lon']),
                            'altitude_ft':  None if a.get('alt_baro') == 'ground' else a.get('alt_baro'),
                            'on_ground':    a.get('alt_baro') == 'ground',
                            'speed_kts':    a.get('gs'),
                            'heading':      a.get('track'),
                            'vertical_fpm': a.get('baro_rate'),
                            'registration': a.get('r'),
                            'aircraft':     a.get('t'),
                            'tried':        tried,
                        }
            except requests.exceptions.RequestException:
                continue

    return {'error': f'No live aircraft found across {len(tried)} ADS-B lookups',
            'tried': tried}

def tool_get_airport_info(iata_code: str) -> dict:
    airports = _load_airports()
    a = airports.get(iata_code.upper().strip())
    if not a:
        return {'error': f'Airport {iata_code} not found'}
    c = a.get('coordinates', {})
    return {'iata': a.get('code'), 'name': a.get('name'),
            'country': a.get('country_code'),
            'latitude': c.get('lat'), 'longitude': c.get('lon')}

TOOLS = {
    'get_flight_details': tool_get_flight_details,
    'get_live_position':  tool_get_live_position,
    'get_airport_info':   tool_get_airport_info,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_flight_details",
        "description": "Look up flight schedule, airline, status, callsign, registration by IATA flight number.",
        "parameters": {"type": "object",
            "properties": {"flight_number": {"type": "string"}},
            "required": ["flight_number"]}}},
    {"type": "function", "function": {
        "name": "get_live_position",
        "description": ("Live ADS-B position. Pass ALL identifiers from get_flight_details: "
                        "callsign (flight_icao), registration (aircraft_reg), icao24 "
                        "(aircraft_icao24), airline_icao, flight_number, flight_iata. "
                        "Auto-generates callsign variants — more inputs = much higher hit rate."),
        "parameters": {"type": "object",
            "properties": {
                "callsign":     {"type": "string"},
                "registration": {"type": "string"},
                "icao24":       {"type": "string"},
                "airline_icao": {"type": "string"},
                "flight_number":{"type": "string"},
                "flight_iata":  {"type": "string"},
            }}}},
    {"type": "function", "function": {
        "name": "get_airport_info",
        "description": "Get airport name and coordinates by IATA code.",
        "parameters": {"type": "object",
            "properties": {"iata_code": {"type": "string"}},
            "required": ["iata_code"]}}},
]

# =====================================================================
# AGENT
# =====================================================================

SYSTEM_PROMPT = """You are a flight tracking assistant.
1. Call get_flight_details with the flight number.
2. Call get_live_position passing ALL of these from step 1's result:
   callsign=flight_icao, registration=aircraft_reg, icao24=aircraft_icao24,
   airline_icao, flight_number, flight_iata. Pass "" for any that are null.
3. Produce a SHORT 2-3 sentence summary: airline, route, status, and live position if found.

Never invent data. If status is 'landed' or 'cancelled', skip get_live_position."""

FRIENDLY_STATUS = {
    'get_flight_details': "Looking up flight schedule…",
    'get_live_position':  "Pinging live ADS-B network…",
    'get_airport_info':   "Fetching airport info…",
}


def _ensure_live_position(tool_results, status_obj=None):
    """Guarantee a thorough live-position lookup after the LLM finishes."""
    flight = tool_results.get('get_flight_details') or {}
    if not flight or flight.get('error'):
        return
    if (flight.get('status') or '').lower() in ('landed', 'cancelled'):
        return

    existing = tool_results.get('get_live_position') or {}
    if existing and not existing.get('error') and existing.get('latitude') is not None:
        return

    if status_obj:
        status_obj.update(label="Running enhanced live-position lookup…")

    live = tool_get_live_position(
        callsign     = str(flight.get('flight_icao')     or ''),
        registration = str(flight.get('aircraft_reg')    or ''),
        icao24       = str(flight.get('aircraft_icao24') or ''),
        airline_icao = str(flight.get('airline_icao')    or ''),
        flight_number= str(flight.get('flight_number')   or ''),
        flight_iata  = str(flight.get('flight_iata')     or ''),
    )
    tool_results['get_live_position'] = live


def run_agent_silent(client, user_query, status_obj=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_query}]
    tool_results = {}
    final_text = "(Max steps reached.)"

    for _ in range(8):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.2)
        except Exception as e:
            final_text = f"LLM error: {e}"
            break

        msg = response.choices[0].message
        msg_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]
        messages.append(msg_dict)

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try: args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError: args = {}
            if status_obj:
                status_obj.update(label=FRIENDLY_STATUS.get(name, f"Calling {name}…"))
            if name in TOOLS:
                try: result = TOOLS[name](**args)
                except Exception as e: result = {'error': f'{type(e).__name__}: {e}'}
            else:
                result = {'error': f'Unknown tool: {name}'}
            tool_results[name] = result
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": name, "content": json.dumps(result)})

    _ensure_live_position(tool_results, status_obj)
    return final_text, tool_results

# =====================================================================
# AESTHETIC UI STYLES
# =====================================================================

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, sans-serif !important;
}
.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.12), transparent),
    radial-gradient(ellipse 60% 50% at 80% 100%, rgba(59,130,246,0.08), transparent),
    #FAFBFF;
}
.block-container { max-width: 1340px; padding-top: 1.5rem; padding-bottom: 4rem; }
#MainMenu, footer, header { visibility: hidden; }

.hero { text-align: center; padding: 28px 0 18px; }
.hero .brand {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 6px 14px; border-radius: 999px;
  background: rgba(99,102,241,0.1); color: #6366F1;
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 2px; margin-bottom: 16px;
}
.hero .brand .live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #10B981; box-shadow: 0 0 8px #10B981;
  animation: pulse 2s infinite;
}
.hero h1 {
  font-size: 3.2rem; font-weight: 800; letter-spacing: -2px;
  background: linear-gradient(135deg, #1E293B 0%, #4F46E5 50%, #06B6D4 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin: 0;
}
.hero p { font-size: 1.05rem; color: #64748B; margin-top: 8px; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes shimmer { 0% { background-position: -1000px 0; } 100% { background-position: 1000px 0; } }

.stForm {
  background: white; border-radius: 18px;
  padding: 8px !important; margin: 8px auto 24px !important;
  box-shadow: 0 10px 40px -10px rgba(99,102,241,0.18),
              0 2px 6px -1px rgba(15,23,42,0.06);
  border: 1px solid rgba(99,102,241,0.1);
  max-width: 720px;
}
.stForm [data-testid="stTextInput"] input {
  border: none !important; background: transparent !important;
  font-size: 1.05rem !important; padding: 14px 16px !important;
  font-weight: 500; color: #0F172A !important;
}
.stForm [data-testid="stTextInput"] input:focus { box-shadow: none !important; }
.stForm [data-testid="stTextInput"] input::placeholder { color: #94A3B8; }
.stForm button[kind="primary"] {
  background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
  border: none !important; border-radius: 12px !important;
  font-weight: 600 !important; height: 48px !important;
  box-shadow: 0 6px 16px -4px rgba(99,102,241,0.4) !important;
  transition: transform 0.15s ease !important;
}
.stForm button[kind="primary"]:hover { transform: translateY(-1px); }

.recent-label {
  font-size: 0.7rem; color: #64748B; text-transform: uppercase;
  letter-spacing: 1.5px; font-weight: 700; margin: 4px 0 8px;
  text-align: center;
}
.recent-bar [data-testid="stHorizontalBlock"] {
  justify-content: center !important; gap: 8px !important;
}
.recent-bar button {
  background: white !important; color: #475569 !important;
  border: 1px solid #E2E8F0 !important; border-radius: 999px !important;
  font-weight: 600 !important; font-size: 0.85rem !important;
  padding: 6px 14px !important; height: auto !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
}
.recent-bar button:hover {
  border-color: #6366F1 !important; color: #4F46E5 !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 10px -2px rgba(99,102,241,0.2) !important;
}

.agent-panel {
  background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
  color: #E2E8F0; border-radius: 22px;
  padding: 26px 24px; position: sticky; top: 1rem;
  box-shadow: 0 24px 60px -20px rgba(15,23,42,0.5),
              0 0 0 1px rgba(99,102,241,0.15) inset;
  overflow: hidden;
}
.agent-panel::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, #6366F1, #06B6D4, transparent);
  animation: shimmer 3s linear infinite;
  background-size: 1000px 100%;
}
.agent-panel .ai-header {
  display: flex; align-items: center; gap: 10px;
  font-size: 0.74rem; font-weight: 800; letter-spacing: 2px;
  text-transform: uppercase; color: #A5B4FC; margin-bottom: 18px;
}
.agent-panel .ai-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #10B981; box-shadow: 0 0 10px #10B981;
  animation: pulse 2s infinite;
}
.agent-panel .ai-query {
  font-size: 0.88rem; color: #CBD5E1;
  background: rgba(99,102,241,0.12); padding: 12px 14px;
  border-radius: 12px; border-left: 3px solid #6366F1;
  margin-bottom: 18px; font-style: italic;
}
.agent-panel .ai-answer {
  font-size: 0.95rem; line-height: 1.65; color: #F8FAFC; font-weight: 400;
}
.agent-panel .ai-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(165,180,252,0.25), transparent);
  margin: 20px 0 14px;
}
.agent-panel .ai-meta-row {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.78rem; color: #94A3B8; padding: 5px 0;
}
.agent-panel .ai-meta-row .v {
  color: #F1F5F9; font-weight: 600;
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.agent-panel .ai-meta-row .v.ok    { color: #34D399; }
.agent-panel .ai-meta-row .v.warn  { color: #FBBF24; }
.agent-panel .ai-section-label {
  font-size: 0.68rem; color: #94A3B8;
  text-transform: uppercase; letter-spacing: 1.5px;
  font-weight: 800; margin-bottom: 10px;
}
.agent-panel .ai-chip {
  display: inline-block; padding: 4px 10px; margin: 2px 4px 2px 0;
  background: rgba(99,102,241,0.15); color: #C7D2FE;
  border-radius: 8px; font-size: 0.72rem; font-weight: 600;
  border: 1px solid rgba(99,102,241,0.25);
}

.fl-hero {
  background:
    radial-gradient(ellipse at top right, rgba(6,182,212,0.3), transparent 60%),
    linear-gradient(135deg, #1E293B 0%, #312E81 50%, #4F46E5 100%);
  color: white; padding: 28px 32px; border-radius: 20px; margin-bottom: 20px;
  box-shadow: 0 24px 50px -15px rgba(79,70,229,0.4);
  position: relative; overflow: hidden;
}
.fl-hero::after {
  content: "✈"; position: absolute;
  font-size: 12rem; right: -20px; top: -30px;
  color: rgba(255,255,255,0.06);
  transform: rotate(-15deg);
  animation: float 4s ease-in-out infinite;
}
.fl-hero .row1 {
  display: flex; justify-content: space-between; align-items: flex-start;
  position: relative; z-index: 2;
}
.fl-hero .flight-no {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -2px;
  font-family: 'JetBrains Mono', monospace;
}
.fl-hero .airline { font-size: 1rem; opacity: 0.85; margin-top: 2px; }
.fl-hero .status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: 1.5px; backdrop-filter: blur(10px);
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.25);
}
.fl-hero .status-pill .sp-dot { width: 6px; height: 6px; border-radius: 50%; }

.route-card {
  background: white; border-radius: 18px;
  padding: 26px 28px; margin-bottom: 20px;
  border: 1px solid #EEF2FF;
  box-shadow: 0 4px 14px -4px rgba(15,23,42,0.06);
}
.route-grid {
  display: grid; grid-template-columns: 1fr auto 1fr;
  gap: 24px; align-items: center;
}
.airport-block .iata-label {
  font-size: 0.68rem; color: #94A3B8;
  text-transform: uppercase; letter-spacing: 2px; font-weight: 700;
}
.airport-block .iata-code {
  font-size: 2.6rem; font-weight: 800; color: #0F172A;
  letter-spacing: -2px; line-height: 1; margin: 4px 0;
  font-family: 'JetBrains Mono', monospace;
}
.airport-block .airport-name {
  font-size: 0.85rem; color: #475569; margin-bottom: 12px; min-height: 32px;
}
.airport-block .time-block {
  background: #F8FAFC; border-radius: 10px;
  padding: 10px 12px; margin-top: 4px;
  border: 1px solid #F1F5F9;
}
.airport-block .time-line {
  display: flex; justify-content: space-between;
  font-size: 0.82rem; padding: 2px 0;
}
.airport-block .time-line .lbl { color: #64748B; }
.airport-block .time-line .val {
  color: #0F172A; font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}
.airport-block.right { text-align: right; }

.route-line {
  display: flex; flex-direction: column; align-items: center;
  padding: 0 20px;
}
.route-line .runway {
  width: 4px; height: 60px;
  background: linear-gradient(180deg, #10B981, #6366F1, #3B82F6);
  border-radius: 2px;
}
.route-line .plane-icon {
  font-size: 1.8rem; color: #6366F1; margin: 8px 0;
  transform: rotate(90deg);
  animation: float 3s ease-in-out infinite;
}
.route-line .duration {
  font-size: 0.72rem; color: #64748B;
  font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; margin-top: 4px;
}

.metric-grid { display: grid; gap: 12px; margin-bottom: 20px; }
.metric-card {
  background: white; border-radius: 14px; padding: 14px 16px;
  border: 1px solid #EEF2FF;
  box-shadow: 0 2px 6px -2px rgba(15,23,42,0.04);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px -6px rgba(99,102,241,0.15);
}
.metric-card .m-icon { font-size: 1.1rem; margin-bottom: 6px; }
.metric-card .m-label {
  font-size: 0.68rem; color: #64748B;
  text-transform: uppercase; letter-spacing: 1px; font-weight: 700;
}
.metric-card .m-value {
  font-size: 1.15rem; font-weight: 800; color: #0F172A;
  margin-top: 2px;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.5px;
}
.metric-card.live-tile {
  background: linear-gradient(135deg, #ECFDF5 0%, #DBEAFE 100%);
  border-color: #A7F3D0;
}
.metric-card.live-tile .m-value { color: #047857; }

.live-section-header {
  display: flex; align-items: center; gap: 10px;
  margin: 20px 0 12px;
}
.live-section-header h3 {
  margin: 0; font-size: 1.05rem; font-weight: 800;
  color: #0F172A; letter-spacing: -0.5px;
}
.live-section-header .live-tag {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px;
  background: linear-gradient(135deg, #EF4444, #F97316);
  color: white; font-size: 0.65rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: 1.5px;
}
.live-section-header .live-tag::before {
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: white; animation: pulse 1.2s infinite;
}

.empty-state {
  text-align: center; padding: 60px 20px; max-width: 540px; margin: 0 auto;
}
.empty-state .empty-plane {
  font-size: 4rem; animation: float 3s ease-in-out infinite;
}
.empty-state h3 {
  font-size: 1.4rem; color: #0F172A; font-weight: 700;
  margin: 16px 0 6px;
}
.empty-state p { color: #64748B; font-size: 0.95rem; }
.empty-state .suggestions {
  display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;
  margin-top: 18px;
}
.empty-state .suggestions code {
  background: white; color: #4F46E5; padding: 5px 12px;
  border-radius: 999px; font-size: 0.85rem; font-weight: 600;
  border: 1px solid #E0E7FF;
  font-family: 'JetBrains Mono', monospace;
}

[data-testid="stExpander"] {
  border-radius: 14px !important; border: 1px solid #EEF2FF !important;
  box-shadow: 0 2px 6px -2px rgba(15,23,42,0.04);
  margin-bottom: 10px;
}
[data-testid="stExpander"] summary {
  font-weight: 600 !important; color: #1E293B !important;
}

.no-live-card {
  background: linear-gradient(135deg, #FEF3C7 0%, #FED7AA 100%);
  border: 1px solid #FCD34D; border-radius: 14px;
  padding: 16px 20px; color: #78350F; font-size: 0.9rem;
  display: flex; gap: 12px; align-items: flex-start;
}
.no-live-card .nl-icon { font-size: 1.6rem; }

.js-plotly-plot .updatemenu-button { cursor: pointer; }
</style>
"""

# =====================================================================
# RENDER
# =====================================================================

def render_animated_map(flight, live):
    airports = _load_airports()
    dep = airports.get(flight.get('dep_iata') or '')
    arr = airports.get(flight.get('arr_iata') or '')
    cur_lat = live.get('latitude'); cur_lon = live.get('longitude')
    if cur_lat is None or cur_lon is None: return

    fig = go.Figure()
    if dep and arr:
        dlat, dlon = dep['coordinates']['lat'], dep['coordinates']['lon']
        alat, alon = arr['coordinates']['lat'], arr['coordinates']['lon']
        fig.add_trace(go.Scattermapbox(
            lat=np.linspace(dlat, alat, 60).tolist(),
            lon=np.linspace(dlon, alon, 60).tolist(),
            mode='lines', line=dict(width=3, color='rgba(99,102,241,0.4)'),
            hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scattermapbox(
            lat=[dlat], lon=[dlon], mode='markers+text',
            marker=dict(size=16, color='#10B981'),
            text=[f"<b>{flight.get('dep_iata','')}</b>"], textposition='top right',
            textfont=dict(size=13, color='#065F46', family='Inter'),
            showlegend=False))
        fig.add_trace(go.Scattermapbox(
            lat=[alat], lon=[alon], mode='markers+text',
            marker=dict(size=16, color='#3B82F6'),
            text=[f"<b>{flight.get('arr_iata','')}</b>"], textposition='top right',
            textfont=dict(size=13, color='#1E3A8A', family='Inter'),
            showlegend=False))
        prog_lats = list(np.linspace(dlat, cur_lat, 20)) + list(np.linspace(cur_lat, alat, 20))[1:]
        prog_lons = list(np.linspace(dlon, cur_lon, 20)) + list(np.linspace(cur_lon, alon, 20))[1:]
    else:
        prog_lats, prog_lons = [cur_lat], [cur_lon]

    fig.add_trace(go.Scattermapbox(
        lat=[cur_lat], lon=[cur_lon], mode='markers',
        marker=dict(size=24, color='#EF4444', opacity=0.3),
        hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scattermapbox(
        lat=[prog_lats[0]], lon=[prog_lons[0]], mode='markers',
        marker=dict(size=14, color='#FBBF24'),
        hovertext=[f"<b>{flight.get('flight_iata')}</b>"], hoverinfo='text',
        showlegend=False))
    moving_idx = len(fig.data) - 1

    fig.frames = [
        go.Frame(data=[go.Scattermapbox(lat=[prog_lats[i]], lon=[prog_lons[i]])],
                 traces=[moving_idx], name=str(i))
        for i in range(len(prog_lats))
    ]

    all_lats = [cur_lat] + ([dep['coordinates']['lat']] if dep else []) + ([arr['coordinates']['lat']] if arr else [])
    all_lons = [cur_lon] + ([dep['coordinates']['lon']] if dep else []) + ([arr['coordinates']['lon']] if arr else [])
    cl = sum(all_lats) / len(all_lats); cln = sum(all_lons) / len(all_lons)
    extent = max(max(all_lats) - min(all_lats), max(all_lons) - min(all_lons), 0.1)
    zoom = 7 if extent < 2 else (5 if extent < 10 else (4 if extent < 30 else 3))

    fig.update_layout(
        mapbox=dict(style='carto-positron', center=dict(lat=cl, lon=cln), zoom=zoom),
        height=480, margin=dict(t=10, b=10, l=0, r=0), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        updatemenus=[
            {'type': 'buttons', 'direction': 'left',
             'x': 0.5, 'y': -0.05, 'xanchor': 'center', 'showactive': False,
             'bgcolor': 'rgba(99,102,241,0.95)', 'bordercolor': '#4F46E5',
             'font': dict(color='white', size=12, family='Inter'),
             'buttons': [
                 {'label': '  ▶  Play  ', 'method': 'animate',
                  'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]},
                 {'label': '  ⏸  Pause  ', 'method': 'animate',
                  'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]},
             ]},
            {'type': 'buttons', 'direction': 'down',
             'x': 1.0, 'y': 1.0, 'xanchor': 'right', 'yanchor': 'top',
             'showactive': False, 'pad': dict(r=8, t=8),
             'bgcolor': 'rgba(255,255,255,0.95)', 'bordercolor': '#E0E7FF',
             'font': dict(color='#4F46E5', size=16, family='Inter'),
             'buttons': [
                 {'label': '＋', 'method': 'relayout',
                  'args': [{'mapbox.zoom': min(zoom + 2, 18)}]},
                 {'label': '−', 'method': 'relayout',
                  'args': [{'mapbox.zoom': max(zoom - 2, 1)}]},
                 {'label': '⟳', 'method': 'relayout',
                  'args': [{'mapbox.zoom': zoom,
                            'mapbox.center': {'lat': cl, 'lon': cln}}]},
             ]},
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'scrollZoom': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toImage', 'toggleSpikelines'],
    })


def render_agent_panel(user_query, final_text, tool_results):
    flight = tool_results.get('get_flight_details') or {}
    live   = tool_results.get('get_live_position') or {}

    tools_used = []
    if 'get_flight_details' in tool_results: tools_used.append('Flight DB')
    if 'get_live_position'  in tool_results: tools_used.append('ADS-B Live')
    if 'get_airport_info'   in tool_results: tools_used.append('Airports')

    if live and not live.get('error') and live.get('latitude'):
        live_v, live_cls = "Tracking", "ok"
    elif live:
        live_v, live_cls = "Not airborne", "warn"
    else:
        live_v, live_cls = "—", ""

    status = (flight.get('status') or '—').title()
    summary_html = (final_text or "No summary available.").replace('\n', '<br>')
    chips_html = "".join(f"<span class='ai-chip'>{t}</span>" for t in tools_used)
    model_short = GROQ_MODEL.split('-')[0].title()

    st.markdown(
        f"""
        <div class='agent-panel'>
          <div class='ai-header'>
            <span class='ai-dot'></span>AI Agent
          </div>
          <div class='ai-query'>"{user_query}"</div>
          <div class='ai-answer'>{summary_html}</div>

          <div class='ai-divider'></div>

          <div class='ai-meta-row'><span>Flight</span>
            <span class='v'>{flight.get('flight_iata') or '—'}</span></div>
          <div class='ai-meta-row'><span>Status</span>
            <span class='v'>{status}</span></div>
          <div class='ai-meta-row'><span>Live tracking</span>
            <span class='v {live_cls}'>{live_v}</span></div>
          <div class='ai-meta-row'><span>Model</span>
            <span class='v'>{model_short}</span></div>

          <div class='ai-divider'></div>

          <div class='ai-section-label'>Tools used</div>
          <div>{chips_html or "<span style='color:#64748B;font-size:0.85rem;'>None</span>"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _calc_duration(dep_iso, arr_iso):
    try:
        d = pd.to_datetime(dep_iso); a = pd.to_datetime(arr_iso)
        if pd.isna(d) or pd.isna(a): return None
        mins = int((a - d).total_seconds() // 60)
        if mins < 0: return None
        h, m = divmod(mins, 60)
        return f"{h}h {m:02d}m"
    except Exception:
        return None


def render_flight_details(tool_results):
    flight = tool_results.get('get_flight_details') or {}
    live   = tool_results.get('get_live_position') or {}

    if flight.get('error'):
        st.error(f"Flight lookup error: {flight['error']}")
        return
    if not flight:
        st.info("No flight data returned.")
        return

    airline = flight.get('airline') or 'Unknown airline'
    iata    = flight.get('flight_iata') or 'N/A'
    status  = (flight.get('status') or 'unknown').lower()
    dot_color, _ = _status_color(status)

    st.markdown(
        f"""
        <div class='fl-hero'>
          <div class='row1'>
            <div>
              <div class='flight-no'>{iata}</div>
              <div class='airline'>{airline}</div>
            </div>
            <div class='status-pill'>
              <span class='sp-dot' style='background:{dot_color};box-shadow:0 0 8px {dot_color};'></span>
              {status}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dep_iata = flight.get('dep_iata') or '?'
    arr_iata = flight.get('arr_iata') or '?'
    dep_airport = flight.get('dep_airport') or ''
    arr_airport = flight.get('arr_airport') or ''
    dep_sch = _fmt_time(flight.get('dep_scheduled'))
    dep_act = _fmt_time(flight.get('dep_actual'))
    arr_sch = _fmt_time(flight.get('arr_scheduled'))
    arr_est = _fmt_time(flight.get('arr_estimated'))
    arr_act = _fmt_time(flight.get('arr_actual'))
    duration = _calc_duration(flight.get('dep_scheduled'), flight.get('arr_scheduled'))

    def build_times(rows):
        return "".join(
            f"<div class='time-line'><span class='lbl'>{lbl}</span>"
            f"<span class='val'>{val}</span></div>"
            for lbl, val in rows if val
        )

    dep_times = build_times([('Scheduled', dep_sch), ('Actual', dep_act)])
    arr_times = build_times([('Scheduled', arr_sch), ('Estimated', arr_est), ('Actual', arr_act)])

    st.markdown(
        f"""
        <div class='route-card'>
          <div class='route-grid'>
            <div class='airport-block'>
              <div class='iata-label'>From</div>
              <div class='iata-code'>{dep_iata}</div>
              <div class='airport-name'>{dep_airport}</div>
              <div class='time-block'>{dep_times or "<div class='time-line'><span class='lbl'>No times</span></div>"}</div>
            </div>
            <div class='route-line'>
              <div class='runway'></div>
              <div class='plane-icon'>✈</div>
              <div class='duration'>{duration or 'En route'}</div>
            </div>
            <div class='airport-block right'>
              <div class='iata-label'>To</div>
              <div class='iata-code'>{arr_iata}</div>
              <div class='airport-name'>{arr_airport}</div>
              <div class='time-block'>{arr_times or "<div class='time-line'><span class='lbl'>No times</span></div>"}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reg    = flight.get('aircraft_reg')  or (live.get('registration') if live else None)
    actype = flight.get('aircraft_type') or (live.get('aircraft')     if live else None)
    cs     = flight.get('flight_icao')   or (live.get('callsign')     if live else None)
    hexid  = flight.get('aircraft_icao24') or (live.get('icao24')     if live else None)

    eq_html = f"""
      <div class='metric-grid' style='grid-template-columns: repeat(4, 1fr);'>
        <div class='metric-card'>
          <div class='m-icon'>🛩</div>
          <div class='m-label'>Aircraft</div>
          <div class='m-value'>{actype or '—'}</div>
        </div>
        <div class='metric-card'>
          <div class='m-icon'>🆔</div>
          <div class='m-label'>Registration</div>
          <div class='m-value'>{reg or '—'}</div>
        </div>
        <div class='metric-card'>
          <div class='m-icon'>📡</div>
          <div class='m-label'>Call sign</div>
          <div class='m-value'>{cs or '—'}</div>
        </div>
        <div class='metric-card'>
          <div class='m-icon'>🔖</div>
          <div class='m-label'>ICAO24</div>
          <div class='m-value'>{(hexid or '—').upper()}</div>
        </div>
      </div>
    """
    st.markdown(eq_html, unsafe_allow_html=True)

    have_adsb = live and not live.get('error') and live.get('latitude') is not None
    have_av   = flight.get('live_latitude') is not None and flight.get('live_longitude') is not None

    if have_adsb or have_av:
        st.markdown(
            "<div class='live-section-header'>"
            "<h3>Live flight data</h3>"
            "<span class='live-tag'>LIVE</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if have_adsb:
            alt = live.get('altitude_ft'); spd = live.get('speed_kts')
            hdg = live.get('heading');     vr  = live.get('vertical_fpm')
            map_live = live
            source_text = f"Live ADS-B data from <b>{live.get('source')}</b>"
        else:
            alt_m = flight.get('live_altitude')
            alt = alt_m * 3.28084 if isinstance(alt_m, (int, float)) else None
            spd_kmh = flight.get('live_speed')
            spd = spd_kmh / 1.852 if isinstance(spd_kmh, (int, float)) else None
            hdg = None; vr = None
            map_live = {'latitude': flight['live_latitude'], 'longitude': flight['live_longitude']}
            source_text = "Live data from <b>AviationStack</b>"

        def fmt(v, suffix):
            if isinstance(v, (int, float)):
                if suffix == ' fpm': return f"{v:+,.0f}{suffix}"
                if suffix == '°':    return f"{v:.0f}{suffix}"
                return f"{v:,.0f}{suffix}"
            return '—'

        live_html = f"""
          <div class='metric-grid' style='grid-template-columns: repeat(4, 1fr);'>
            <div class='metric-card live-tile'>
              <div class='m-icon'>⬆</div><div class='m-label'>Altitude</div>
              <div class='m-value'>{fmt(alt, ' ft')}</div></div>
            <div class='metric-card live-tile'>
              <div class='m-icon'>💨</div><div class='m-label'>Speed</div>
              <div class='m-value'>{fmt(spd, ' kts')}</div></div>
            <div class='metric-card live-tile'>
              <div class='m-icon'>🧭</div><div class='m-label'>Heading</div>
              <div class='m-value'>{fmt(hdg, '°')}</div></div>
            <div class='metric-card live-tile'>
              <div class='m-icon'>📈</div><div class='m-label'>Vertical</div>
              <div class='m-value'>{fmt(vr, ' fpm')}</div></div>
          </div>
        """
        st.markdown(live_html, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem;color:#64748B;margin:-8px 0 10px;'>{source_text}</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.75rem;color:#94A3B8;margin-bottom:4px;'>"
            "Use <b>＋ − ⟳</b> in the top-right of the map, or scroll-wheel to zoom."
            "</div>",
            unsafe_allow_html=True,
        )
        render_animated_map(flight, map_live)
    else:
        tried = (live or {}).get('tried', [])
        tried_html = ""
        if tried:
            tried_html = (
                "<details style='margin-top:10px;'>"
                "<summary style='cursor:pointer;font-size:0.8rem;color:#92400E;font-weight:600;'>"
                f"Show {len(tried)} lookup attempts</summary>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                "color:#78350F;margin-top:8px;max-height:160px;overflow-y:auto;"
                "background:rgba(255,255,255,0.6);padding:10px;border-radius:8px;"
                "line-height:1.6;'>"
                + "<br>".join(tried) +
                "</div></details>"
            )
        st.markdown(
            "<div class='no-live-card'>"
            "<div class='nl-icon'>🛬</div>"
            "<div style='flex:1;'><b>Live position not available.</b><br>"
            "<span style='font-size:0.85rem;'>"
            "Aircraft may be on the ground, not yet departed, or out of ADS-B receiver coverage."
            "</span>"
            f"{tried_html}"
            "</div></div>",
            unsafe_allow_html=True,
        )


# =====================================================================
# MAIN
# =====================================================================

st.set_page_config(page_title="Flight Tracker", layout="wide", page_icon="✈")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class='hero'>
      <div class='brand'>
        <span class='live-dot'></span>
        AI-Powered Live Flight Tracker
      </div>
      <h1>Where is your flight?</h1>
      <p>Ask in plain English. Powered by ReAct agents and global ADS-B mirrors.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.error("**GROQ_API_KEY not set.** Add it to Colab Secrets and re-run your launcher cell.")
    st.stop()
if not AVIATIONSTACK_ACCESS_KEY:
    st.warning("AVIATIONSTACK_API_KEY not set — flight lookups will fail.")

groq_client = Groq(api_key=GROQ_API_KEY)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'pending_query' not in st.session_state:
    st.session_state.pending_query = None

with st.form("flight_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        typed = st.text_input(
            "query",
            placeholder="🔍   Try: 'Where is BA249?'   or   'Track EK205'",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Search", use_container_width=True, type="primary")

query_to_run = None
if submitted and typed.strip():
    query_to_run = typed.strip()
elif st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None

if query_to_run:
    with st.status(f"Searching for **{query_to_run}**…", expanded=False) as status:
        final_text, tool_results = run_agent_silent(groq_client, query_to_run, status)
        status.update(label="Done", state="complete", expanded=False)

    st.session_state.history.insert(0, {
        'query':        query_to_run,
        'final_text':   final_text,
        'tool_results': tool_results,
    })
    st.session_state.history = st.session_state.history[:10]

if st.session_state.history:
    st.markdown("<div class='recent-label'>Recent searches</div>", unsafe_allow_html=True)
    st.markdown("<div class='recent-bar'>", unsafe_allow_html=True)
    n_chips = min(len(st.session_state.history), 6)
    chip_cols = st.columns(n_chips + 1)
    for i, h in enumerate(st.session_state.history[:n_chips]):
        flt = (h['tool_results'].get('get_flight_details') or {}).get('flight_iata') or h['query']
        if chip_cols[i].button(f"✈ {flt}", key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_query = h['query']
            st.rerun()
    if chip_cols[-1].button("✕ Clear", key="clear_hist", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    latest = st.session_state.history[0]
    left, right = st.columns([1, 2], gap="large")
    with left:
        render_agent_panel(latest['query'], latest['final_text'], latest['tool_results'])
    with right:
        render_flight_details(latest['tool_results'])

    if len(st.session_state.history) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.78rem;color:#64748B;font-weight:800;"
            "text-transform:uppercase;letter-spacing:1.5px;margin:14px 0 10px;'>"
            "Previous results</div>",
            unsafe_allow_html=True,
        )
        for idx, item in enumerate(st.session_state.history[1:], start=1):
            flt = (item['tool_results'].get('get_flight_details') or {}).get('flight_iata') or item['query']
            status_v = (item['tool_results'].get('get_flight_details') or {}).get('status') or '—'
            with st.expander(f"✈  {flt}  ·  {item['query']}  ·  {status_v}"):
                l2, r2 = st.columns([1, 2], gap="large")
                with l2:
                    render_agent_panel(item['query'], item['final_text'], item['tool_results'])
                with r2:
                    render_flight_details(item['tool_results'])
else:
    st.markdown(
        """
        <div class='empty-state'>
          <div class='empty-plane'>✈</div>
          <h3>Enter a flight number above to begin</h3>
          <p>Track live position, schedules, and routes for any commercial flight worldwide.</p>
          <div class='suggestions'>
            <code>BA249</code>
            <code>EK205</code>
            <code>Where is QF11?</code>
            <code>Track SQ22</code>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
