"""
Unified Feeling → Meal / Music / Entertainment Streamlit App
- Enter feeling text
- Optionally provide entertainment likes (genres, shows, actors)
- Choose outputs (meal, music, entertainment)
- App uses a small rulebook + OpenAI LLM to generate structured recommendations + reasoning

Install:
    pip install -r requirements.txt

Run (PowerShell):
    streamlit run app.py

Notes:
- Enter your OpenAI API key in the sidebar.
- The app uses the OpenAI Responses API via the official `openai` Python client.
"""

import streamlit as st
from openai import OpenAI
import json
import os
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Feeling → Meal/Music/Entertainment", layout="wide")
st.title("🎭 Feeling → Meal · Music · Entertainment")
st.write("Describe your feeling and get a meal (with recipe), a music playlist, and entertainment picks (movie/anime/series) — all aligned with your feeling and preferences.")

# -------------------------
# Sidebar: API & options
# -------------------------
provider = st.sidebar.selectbox("AI Provider", ["Google Gemini", "OpenAI"], index=0)
api_key = st.sidebar.text_input(f"{provider} API Key", type="password", value=os.getenv(f"{provider.upper().replace(' ', '_')}_API_KEY", ""))
if provider == "OpenAI":
    model_choice = st.sidebar.selectbox("Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-o1"], index=0)
else:
    model_choice = st.sidebar.selectbox("Model", ["gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash"], index=0)
mock_mode = st.sidebar.checkbox("Developer mock/test mode (no API key)", value=False, help="When enabled the app returns canned outputs for development without calling any LLM.")
st.sidebar.markdown("---")
st.sidebar.write("Mapping strictness controls how much the system adheres to the built-in feeling rules vs LLM nuance.")
strictness = st.sidebar.slider("Mapping Strictness", 0, 100, 70)
st.sidebar.write("Higher = more rule-driven anchors; Lower = more LLM freedom.")
st.sidebar.markdown("---")
st.sidebar.write("Tip: Add entertainment likes (genres, shows, actors) to bias recommendations.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Made by Team FLAME FOX**")

if not api_key and not mock_mode:
    st.warning(f"Please enter your {provider} API key in the sidebar to use the app (or enable mock/test mode).")
    st.stop()

client = None
if not mock_mode:
    if provider == "Google Gemini":
        if not GEMINI_AVAILABLE:
            st.error("google-generativeai package not installed. Run: pip install google-generativeai")
            st.stop()
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model_choice)
    else:
        client = OpenAI(api_key=api_key)

# -------------------------
# Human-designed feeling -> anchor rules (editable)
# -------------------------
MOOD_RULES = {
    "energy": {
        "low": {
            "meal": ["warm, light, easy-to-digest (soups, porridges)", "low-prep comfort food"],
            "music": ["slow tempo, ambient, acoustic"],
            "entertainment": ["gentle pacing, feel-good dramas, slice-of-life"]
        },
        "medium": {
            "meal": ["balanced bowls (protein + veg + grain)", "simple pastas, stir-fries"],
            "music": ["mid-tempo pop/indie, chill electronic"],
            "entertainment": ["dramas with moderate pacing, comedies"]
        },
        "high": {
            "meal": ["energizing foods: spicy, crunchy, protein-rich"],
            "music": ["upbeat, fast-tempo genres: dance, rock"],
            "entertainment": ["action, thrillers, fast-paced series"]
        }
    },
    "valence": {
        "positive": {
            "meal": ["bright flavors, citrus, fresh salads"],
            "music": ["major-key upbeat songs"],
            "entertainment": ["feel-good, visually joyful films / anime"]
        },
        "neutral": {
            "meal": ["familiar, low-risk comfort food"],
            "music": ["ambient, lo-fi, focus playlists"],
            "entertainment": ["documentaries, slice-of-life, low-intensity shows"]
        },
        "negative": {
            "meal": ["grounding comfort food (stews, mashed textures)"],
            "music": ["soft acoustic, soothing instrumental"],
            "entertainment": ["cathartic or uplifting films; healing 'iyashikei' anime"]
        }
    },
    "stress": {
        "low": {
            "meal": ["something creative or celebratory"],
            "music": ["playful or exploratory playlists"],
            "entertainment": ["quirky or thoughtful films"]
        },
        "high": {
            "meal": ["very low-prep, familiar dishes to reduce cognitive load"],
            "music": ["calming steady playlists"],
            "entertainment": ["comfort shows, short episodic series"]
        }
    }
}

# -------------------------
# Helpers: call LLM
# -------------------------
def call_llm(prompt: str, model: str = None, max_tokens: int = 700):
    model = model or model_choice
    try:
        if provider == "Google Gemini":
            # Use Gemini API
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
            )
            return response.text
        else:
            # Use OpenAI API
            resp = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens)
            # `resp.output_text` is a convenience in some client versions — fallback to content traversal
            if hasattr(resp, 'output_text'):
                return resp.output_text
            # try to extract text from choices / output
            if hasattr(resp, 'output') and isinstance(resp.output, list) and len(resp.output) > 0:
                # join any text content
                parts = []
                for item in resp.output:
                    if isinstance(item, dict) and item.get('content'):
                        for c in item['content']:
                            if c.get('type') == 'output_text' and c.get('text'):
                                parts.append(c.get('text'))
                if parts:
                    return "\n".join(parts)
            return str(resp)
    except Exception as e:
        return f"ERROR: {e}"

# -------------------------
# Feeling analysis
# -------------------------
def heuristic_mood_parse(text: str):
    t = text.lower()
    energy = "medium"
    valence = "neutral"
    stress = "low"
    sensory = ""

    if any(w in t for w in ["tired", "drained", "lethargic", "sleepy", "exhausted"]):
        energy = "low"
    if any(w in t for w in ["energetic", "pumped", "hyper", "excited", "restless"]):
        energy = "high"

    if any(w in t for w in ["happy", "joyful", "optimistic", "good", "cheerful"]):
        valence = "positive"
    if any(w in t for w in ["sad", "down", "depressed", "lonely", "angry", "upset"]):
        valence = "negative"

    if any(w in t for w in ["anxious", "stressed", "overwhelmed", "nervous", "tense"]):
        stress = "high"

    # sensory hints
    if any(w in t for w in ["warm", "cozy", "hot", "cold", "refreshing", "spicy", "sweet"]):
        sensory = ", ".join([w for w in ["warm", "cozy", "hot", "cold", "refreshing", "spicy", "sweet"] if w in t])

    return {"energy": energy, "valence": valence, "stress": stress, "sensory": sensory}

# LLM fallback to parse feeling in ambiguous cases or to refine heuristics
MOOD_PARSE_PROMPT = '''
You are a concise assistant that converts a short feeling description into three axes.
Return JSON only:

{{
 "energy": "low|medium|high",
 "valence": "negative|neutral|positive",
 "stress": "low|high",
 "sensory": "comma-separated short keywords or empty"
}}

Feeling description:
"""{mood_text}"""
'''

def parse_mood(mood_text: str):
    parsed = heuristic_mood_parse(mood_text)
    # If the input contains conjunctions or 'but' or the strictness is low, ask LLM to be sure
    ambiguous = ("but" in mood_text.lower()) or ("," in mood_text and "but" in mood_text.lower())
    if ambiguous or strictness < 90:
        prompt = MOOD_PARSE_PROMPT.format(mood_text=mood_text)
        out = call_llm(prompt)
        try:
            j = json.loads(out)
            # sanitize values
            for k in ("energy", "valence", "stress", "sensory"):
                if k in j and j[k]:
                    parsed[k] = j[k]
        except Exception:
            # keep heuristic if parsing fails, but attach raw output to notes
            parsed["parse_debug"] = out[:800]
    return parsed

# -------------------------
# Anchors: gather suggested anchor texts from rules
# -------------------------
def anchors_for(parsed, domain):
    anchors = []
    for axis in ("energy", "valence", "stress"):
        val = parsed.get(axis)
        try:
            domain_anchors = MOOD_RULES[axis][val][domain]
            anchors.extend(domain_anchors)
        except Exception:
            continue
    # include sensory if present
    if parsed.get("sensory"):
        anchors.append(parsed["sensory"])
    # dedupe and limit
    seen = []
    for a in anchors:
        if a not in seen:
            seen.append(a)
    return seen[:6]

# -------------------------
# Prompts for generators
# -------------------------
MEAL_PROMPT = """
You are a friendly culinary assistant. Input:
- feeling_text: {mood}
- feeling_anchors: {anchors}
- constraint: keep prep time under 30 minutes unless the feeling explicitly calls for slow cooking.
 - dietary: {dietary}
 - time_of_day: {time_of_day}

Produce JSON ONLY:
{{
 "meal": "<short dish title>",
 "reasoning": "<1-2 short sentences why this fits (mention energy/valence/stress)>",
 "recipe": {{
     "time": "<approx minutes>",
     "ingredients": ["...","..."],
     "steps": ["step 1..","step 2.."]
 }}
}}
"""

MUSIC_PROMPT = """
You are a concise music recommender. Input:
- feeling_text: {mood}
- feeling_anchors: {anchors}

Produce JSON ONLY:
{{
 "playlist": "<short playlist title>",
 "tracks": ["Artist - Song", "Artist - Song", "..."] , 
 "reasoning": "<one sentence>"
}}
"""

ENTERTAINMENT_PROMPT = """
You are a helpful entertainment recommender.
Input:
- feeling_text: {mood}
- feeling_anchors: {anchors}
- user_likes: {likes}   # a short user preference string; can be empty

Produce JSON ONLY: a list of three recommendations (movie / anime / series) ranked 1..3.
Each recommendation must include:
- title (title + year if available)
- type ("movie" / "anime" / "series")
- one-sentence synopsis (no spoilers)
- two-sentence reasoning why it suits the feeling and how it aligns with user_likes (if provided)
- suggested viewing context (e.g., "alone, with friends, snack suggestions, time of day")

Return JSON array: [ {{...}}, {{...}}, {{...}} ]
"""

# -------------------------
# Generator functions
# -------------------------
def generate_meal(mood_text, parsed):
    anchors = anchors_for(parsed, "meal")
    anchor_text = "; ".join(anchors) if anchors else "general comforting meal"
    prompt = MEAL_PROMPT.format(mood=mood_text, anchors=anchor_text, dietary=parsed.get('dietary',''), time_of_day=parsed.get('time_of_day','any'))
    if mock_mode:
        # Return a canned meal aligned to parsed axes
        return {
            "meal": "Comforting Tomato & Basil Soup",
            "reasoning": "Warm, low-effort, and soothing — fits low energy and high stress.",
            "recipe": {
                "time": "25",
                "ingredients": ["1 can crushed tomatoes", "1 small onion, diced", "1 cup vegetable broth", "2 cloves garlic", "fresh basil", "salt & pepper"],
                "steps": ["Sauté onion and garlic until soft.", "Add tomatoes and broth, simmer 15 min.", "Blend until smooth, stir in basil and season to taste."]
            }
        }
    out = call_llm(prompt)
    try:
        return json.loads(out)
    except Exception:
        return {"meal": "Error generating meal", "reasoning": out[:600], "recipe": {"time": "", "ingredients": [], "steps": []}}


def generate_music(mood_text, parsed):
    anchors = anchors_for(parsed, "music")
    anchor_text = "; ".join(anchors) if anchors else "feeling-aligned music"
    prompt = MUSIC_PROMPT.format(mood=mood_text, anchors=anchor_text)
    if mock_mode:
        return {
            "playlist": "Mellow & Warm",
            "tracks": ["Iron & Wine - Naked As We Came", "Bon Iver - Holocene", "Norah Jones - Sunrise", "Sufjan Stevens - Should Have Known Better", "Explosions in the Sky - Your Hand in Mine", "Ólafur Arnalds - Near Light"],
            "reasoning": "Slow tempo and acoustic textures to match low energy and soothe stress."
        }
    out = call_llm(prompt)
    try:
        return json.loads(out)
    except Exception:
        return {"playlist": "Error", "tracks": [], "reasoning": out[:600]}


def generate_entertainment(mood_text, parsed, user_likes):
    anchors = anchors_for(parsed, "entertainment")
    anchor_text = "; ".join(anchors) if anchors else "feeling-aligned entertainment"
    likes_text = user_likes if user_likes else ""
    prompt = ENTERTAINMENT_PROMPT.format(mood=mood_text, anchors=anchor_text, likes=likes_text)
    if mock_mode:
        return [
            {"title": "Whisper of the Heart (1995)", "type": "anime", "synopsis": "A gentle coming-of-age story about creativity and young optimism.", "reasoning": "Soft, uplifting pacing suits neutral-to-positive valence and low stress; aligns with 'hopeful' likes.", "context": "Alone or with a close friend; evening; light snacks."},
            {"title": "The Secret Life of Walter Mitty (2013)", "type": "movie", "synopsis": "An ordinary man goes on an extraordinary, visually uplifting journey.", "reasoning": "Uplifting and visually rich, good for boosting positive valence and energy.", "context": "Afternoon, with coffee or a light meal."},
            {"title": "Ted Lasso (2020- )", "type": "series", "synopsis": "A warm, optimistic coach changes lives with kindness and humor.", "reasoning": "Feel-good series that soothes stress and supports positive emotions.", "context": "Evening binge; with friends or solo for comfort."}
        ]
    out = call_llm(prompt, max_tokens=900)
    try:
        j = json.loads(out)
        # Expecting list
        if isinstance(j, dict):
            # Some models return an object; wrap
            return [j]
        return j
    except Exception:
        return [{"title": "Error generating entertainment", "type": "error", "synopsis": out[:800], "reasoning": "", "context": ""}]

# -------------------------
# UI: Inputs
# -------------------------
st.subheader("Input")
mood_input = st.text_area("Describe your current feeling (e.g., 'slightly drained but hopeful')", height=120)
user_likes = st.text_input("Entertainment likes (optional) — genres, shows, actors, or examples", placeholder="e.g., 'romcoms, Miyazaki, sci-fi, The Crown'")
dietary = st.text_input("Dietary restrictions (optional)", placeholder="e.g., 'vegetarian, gluten-free, nut allergy'")
time_of_day = st.selectbox("Time of day", ["any","breakfast","lunch","dinner","late night"], index=0)

# Example feeling presets for quick testing
with st.expander("Example feelings"):
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Cozy & tired"):
        mood_input = "tired, want something warm and cozy, low energy"
    if col_b.button("Pumped & excited"):
        mood_input = "energetic and excited, ready to go out and dance"
    if col_c.button("Anxious but hopeful"):
        mood_input = "anxious and overwhelmed but a little hopeful; need calming and uplifting things"
col1, col2, col3 = st.columns(3)
with col1:
    want_meal = st.checkbox("Meal + Recipe", value=True)
with col2:
    want_music = st.checkbox("Music Playlist", value=True)
with col3:
    want_ent = st.checkbox("Entertainment Picks (movie/anime/series)", value=True)

if st.button("Generate Recommendations"):
    if not mood_input.strip():
        st.error("Please enter a feeling description.")
        st.stop()

    with st.spinner("Parsing feeling..."):
        parsed = parse_mood(mood_input)

    # attach dietary/time to parsed for prompts
    parsed['dietary'] = dietary
    parsed['time_of_day'] = time_of_day

    st.markdown("### 🧭 Parsed Feeling Axes")
    st.write(parsed)

    # Generate outputs
    # session state for cached outputs so regenerate works per-domain
    if 'meal_obj' not in st.session_state:
        st.session_state['meal_obj'] = None
    if 'music_obj' not in st.session_state:
        st.session_state['music_obj'] = None
    if 'ent_list' not in st.session_state:
        st.session_state['ent_list'] = None

    if want_meal:
        colm1, colm2 = st.columns([8,1])
        with colm1:
            st.markdown("## 🍽 Meal Recommendation")
        with colm2:
            if st.button("Regenerate Meal"):
                st.session_state['meal_obj'] = None
        if st.session_state.get('meal_obj') is None:
            with st.spinner("Generating meal..."):
                st.session_state['meal_obj'] = generate_meal(mood_input, parsed)
        meal_obj = st.session_state['meal_obj']
        st.markdown("## 🍽 Meal Recommendation")
        if meal_obj.get("meal"):
            st.markdown(f"**{meal_obj['meal']}**")
            st.markdown(f"**Reasoning:** {meal_obj.get('reasoning','')}")
            rec = meal_obj.get("recipe", {})
            st.markdown(f"**Estimated time:** {rec.get('time','')}")
            st.markdown("**Ingredients:**")
            for ing in rec.get("ingredients", []):
                st.write(f"- {ing}")
            st.markdown("**Steps:**")
            for i, s in enumerate(rec.get("steps", []), start=1):
                st.write(f"{i}. {s}")
        else:
            st.write(meal_obj)

    if want_music:
        colm1, colm2 = st.columns([8,1])
        with colm1:
            st.markdown("## 🎧 Music / Playlist")
        with colm2:
            if st.button("Regenerate Music"):
                st.session_state['music_obj'] = None
        if st.session_state.get('music_obj') is None:
            with st.spinner("Generating playlist..."):
                st.session_state['music_obj'] = generate_music(mood_input, parsed)
        music_obj = st.session_state['music_obj']
        if music_obj.get("playlist"):
            st.markdown(f"**{music_obj['playlist']}**")
            st.markdown("**Tracks:**")
            for t in music_obj.get("tracks", []):
                st.write(f"- {t}")
            st.markdown(f"**Reasoning:** {music_obj.get('reasoning','')}")
        else:
            st.write(music_obj)

    if want_ent:
        colm1, colm2 = st.columns([8,1])
        with colm1:
            st.markdown("## 🎬 Entertainment Picks")
        with colm2:
            if st.button("Regenerate Entertainment"):
                st.session_state['ent_list'] = None
        if st.session_state.get('ent_list') is None:
            with st.spinner("Generating entertainment picks..."):
                st.session_state['ent_list'] = generate_entertainment(mood_input, parsed, user_likes)
        ent_list = st.session_state['ent_list']
        for idx, rec in enumerate(ent_list, start=1):
            st.markdown(f"### {idx}. {rec.get('title','(no title)')} — {rec.get('type','')}")
            st.write(f"Synopsis: {rec.get('synopsis','')}")
            st.markdown(f"**Why it fits:** {rec.get('reasoning','')}")
            st.write(f"Viewing context: {rec.get('context','')}")
            st.markdown("---")

    st.write("Tip: adjust 'Mapping Strictness' in the sidebar to make outputs more rule-guided (higher) or more LLM-driven (lower).")

# Footer
st.markdown("---")
st.write("Built with ❤️ — a unified feeling→recommendation demo using OpenAI models. Edit the rulebook (MOOD_RULES) and prompts inline to customize behavior.")
st.markdown("**Made by Team FLAME FOX**")

# End
