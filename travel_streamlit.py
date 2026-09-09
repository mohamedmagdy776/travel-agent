"""
Travel Agent - Streamlit UI (Arabic + English)
===============================================
Run locally:  streamlit run travel_streamlit.py
Deploy free:  push to GitHub → streamlit.io/cloud → connect repo → deploy
"""

import streamlit as st
import os
from datetime import date, timedelta

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Planner | مخطط السفر",
    page_icon="✈️",
    layout="wide",
)

# ─── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #1a56db;
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton > button:hover { background-color: #1e429f; }
    .rtl { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ─── Language Selector ─────────────────────────────────────────
col_lang1, col_lang2, col_lang3 = st.columns([1, 1, 4])
with col_lang1:
    if st.button("🇬🇧 English"):
        st.session_state["lang"] = "en"
with col_lang2:
    if st.button("🇸🇦 العربية"):
        st.session_state["lang"] = "ar"

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

lang = st.session_state["lang"]
is_ar = lang == "ar"

# ─── Text Content ──────────────────────────────────────────────
T = {
    "title":            {"en": "✈️ AI Travel Planner",       "ar": "✈️ مخطط السفر الذكي"},
    "subtitle":         {"en": "Tell us where you want to go and we'll plan everything for you.",
                         "ar": "قولنا عايز تروح فين وإحنا هنخطط كل حاجة عنك."},
    "setup":            {"en": "⚙️ Setup",                   "ar": "⚙️ الإعدادات"},
    "openai_key":       {"en": "OpenAI API Key",             "ar": "مفتاح OpenAI"},
    "tavily_key":       {"en": "Tavily API Key",             "ar": "مفتاح Tavily"},
    "what_we_do":       {"en": "**What we do for you:**",    "ar": "**إيه اللي بنعمله عشانك:**"},
    "where":            {"en": "Where are you going?",       "ar": "عايز تروح فين؟"},
    "from":             {"en": "Flying from",                "ar": "السفر من"},
    "from_ph":          {"en": "e.g. Cairo, Egypt",          "ar": "مثلاً: القاهرة، مصر"},
    "to":               {"en": "Destination",                "ar": "الوجهة"},
    "to_ph":            {"en": "e.g. Tokyo, Japan",          "ar": "مثلاً: طوكيو، اليابان"},
    "when":             {"en": "When?",                      "ar": "إمتى؟"},
    "departure":        {"en": "Departure date",             "ar": "تاريخ السفر"},
    "days":             {"en": "Number of days",             "ar": "عدد الأيام"},
    "travelers":        {"en": "Travelers",                  "ar": "عدد المسافرين"},
    "budget_title":     {"en": "Your budget & preferences",  "ar": "الميزانية والتفضيلات"},
    "budget":           {"en": "Total budget",               "ar": "الميزانية الكلية"},
    "style":            {"en": "Travel style",               "ar": "نوع السفر"},
    "styles":           {"en": ["Budget backpacker", "Mid-range comfort", "Luxury"],
                         "ar": ["اقتصادي", "متوسط", "فاخر"]},
    "interests":        {"en": "What do you love?",          "ar": "إيه اللي بتحبه؟"},
    "interest_opts":    {"en": ["Food & restaurants", "History & culture", "Nature & hiking",
                                 "Shopping", "Nightlife", "Beaches", "Museums & art",
                                 "Anime & pop culture", "Adventure sports", "Photography spots"],
                         "ar": ["أكل ومطاعم", "تاريخ وثقافة", "طبيعة وتسلق",
                                "تسوق", "حياة ليلية", "شواطئ", "متاحف وفن",
                                "أنيمي وثقافة شعبية", "رياضات مغامرة", "أماكن تصوير"]},
    "default_interests":{"en": ["Food & restaurants", "History & culture"],
                         "ar": ["أكل ومطاعم", "تاريخ وثقافة"]},
    "nationality":      {"en": "Your nationality",            "ar": "جنسيتك"},
    "nationality_ph":   {"en": "e.g. Egyptian, American, British...", "ar": "مثلاً: مصري، أمريكي، بريطاني..."},
    "currency":         {"en": "Preferred currency",            "ar": "العملة المفضلة"},
    "currencies":       {"en": ["USD 🇺🇸", "EGP 🇪🇬", "EUR 🇪🇺", "GBP 🇬🇧", "SAR 🇸🇦", "AED 🇦🇪", "JPY 🇯🇵", "TRY 🇹🇷"],
                         "ar": ["جنيه مصري EGP 🇪🇬", "دولار USD 🇺🇸", "يورو EUR 🇪🇺", "جنيه إسترليني GBP 🇬🇧", "ريال سعودي SAR 🇸🇦", "درهم إماراتي AED 🇦🇪", "ين ياباني JPY 🇯🇵", "ليرة تركية TRY 🇹🇷"]},
    "special":          {"en": "Anything else? (optional)",  "ar": "في حاجة تانية؟ (اختياري)"},
    "special_ph":       {"en": "e.g. I have a shellfish allergy. I prefer boutique hotels.",
                         "ar": "مثلاً: عندي حساسية من المأكولات البحرية. بفضل الفنادق الصغيرة."},
    "btn":              {"en": "🔍 Plan My Trip",            "ar": "🔍 خطط رحلتي"},
    "no_from":          {"en": "Please enter your departure city.",   "ar": "من فضلك ادخل مدينة السفر."},
    "no_to":            {"en": "Please enter your destination.",      "ar": "من فضلك ادخل الوجهة."},
    "no_key":           {"en": "Please enter your OpenAI API key.",   "ar": "من فضلك ادخل مفتاح OpenAI."},
    "planning":         {"en": "Planning your trip to",      "ar": "بيتم تخطيط رحلتك إلى"},
    "days_label":       {"en": "days",                       "ar": "يوم"},
    "steps": {
        "en": ["📋 Building your travel plan...", "✈️ Searching for flights...",
               "🏨 Finding hotels...", "🛂 Checking visa requirements...",
               "🌤️ Checking weather...", "🎯 Researching activities...",
               "📍 Finding the best places...", "💰 Calculating your budget...",
               "✍️ Writing your itinerary...", "🔍 Final review..."],
        "ar": ["📋 بيبني خطة رحلتك...", "✈️ بيدور على رحلات طيران...",
               "🏨 بيدور على فنادق...", "🛂 بيشوف متطلبات التأشيرة...",
               "🌤️ بيشوف الطقس...", "🎯 بيدور على أنشطة...",
               "📍 بيدور على أحسن الأماكن...", "💰 بيحسب الميزانية...",
               "✍️ بيكتب برنامجك...", "🔍 مراجعة أخيرة..."],
    },
    "ready":            {"en": "✅ Your itinerary is ready!", "ar": "✅ برنامجك جاهز!"},
    "dest_label":       {"en": "📍 Destination",             "ar": "📍 الوجهة"},
    "duration_label":   {"en": "📅 Duration",                "ar": "📅 المدة"},
    "travelers_label":  {"en": "👥 Travelers",               "ar": "👥 المسافرون"},
    "budget_label":     {"en": "💰 Budget",                  "ar": "💰 الميزانية"},
    "tab1":             {"en": "📄 Full Itinerary",          "ar": "📄 البرنامج الكامل"},
    "tab2":             {"en": "✈️ Flights & Booking",       "ar": "✈️ طيران وحجز"},
    "tab3":             {"en": "🛂 Visa & Weather",          "ar": "🛂 تأشيرة وطقس"},
    "tab4":             {"en": "💰 Budget",                  "ar": "💰 الميزانية"},
    "flights":          {"en": "✈️ Flights",                 "ar": "✈️ رحلات الطيران"},
    "hotels":           {"en": "🏨 Hotels",                  "ar": "🏨 الفنادق"},
    "visa":             {"en": "🛂 Visa Requirements",       "ar": "🛂 متطلبات التأشيرة"},
    "weather":          {"en": "🌤️ Weather",                 "ar": "🌤️ الطقس"},
    "budget_break":     {"en": "💰 Budget Breakdown",        "ar": "💰 تفاصيل الميزانية"},
    "download":         {"en": "📥 Download itinerary",      "ar": "📥 تحميل البرنامج"},
    "error":            {"en": "Something went wrong. Please try again.",
                         "ar": "في مشكلة حصلت. حاول تاني."},
    "sidebar_items":    {
        "en": ["✈️ Search real flights", "🏨 Find hotels in your budget",
               "🛂 Check visa requirements", "🌤️ Check the weather",
               "🎯 Recommend activities", "📍 Find best places",
               "💰 Calculate your budget", "📋 Write full itinerary"],
        "ar": ["✈️ بيدور على رحلات حقيقية", "🏨 بيلاقي فنادق في ميزانيتك",
               "🛂 بيشوف التأشيرة", "🌤️ بيشوف الطقس",
               "🎯 بيقترح أنشطة", "📍 بيلاقي أحسن الأماكن",
               "💰 بيحسب ميزانيتك", "📋 بيكتب برنامجك كامل"],
    },
}

def t(key):
    return T[key][lang]

# ─── Header ────────────────────────────────────────────────────
if is_ar:
    st.markdown(f'<div class="rtl"><h1>{t("title")}</h1><p>{t("subtitle")}</p></div>',
                unsafe_allow_html=True)
else:
    st.title(t("title"))
    st.markdown(t("subtitle"))

st.markdown("---")

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header(t("setup"))
    openai_key = st.text_input(t("openai_key"), type="password")
    tavily_key = st.text_input(t("tavily_key"), type="password")
    st.markdown("---")
    st.markdown(t("what_we_do"))
    for item in t("sidebar_items"):
        st.markdown(item)

# ─── Form ──────────────────────────────────────────────────────
with st.form("travel_form"):
    if is_ar:
        st.markdown(f'<div class="rtl"><h3>{t("where")}</h3></div>', unsafe_allow_html=True)
    else:
        st.subheader(t("where"))

    col1, col2 = st.columns(2)
    with col1:
        from_city = st.text_input(t("from"), placeholder=t("from_ph"))
    with col2:
        to_city = st.text_input(t("to"), placeholder=t("to_ph"))

    nationality = st.text_input(t("nationality"), placeholder=t("nationality_ph"))

    if is_ar:
        st.markdown(f'<div class="rtl"><h3>{t("when")}</h3></div>', unsafe_allow_html=True)
    else:
        st.subheader(t("when"))

    col3, col4, col5 = st.columns(3)
    with col3:
        departure = st.date_input(t("departure"),
                                   value=date.today() + timedelta(days=30),
                                   min_value=date.today())
    with col4:
        num_days = st.number_input(t("days"), min_value=1, max_value=30, value=7)
    with col5:
        num_travelers = st.number_input(t("travelers"), min_value=1, max_value=10, value=1)

    if is_ar:
        st.markdown(f'<div class="rtl"><h3>{t("budget_title")}</h3></div>', unsafe_allow_html=True)
    else:
        st.subheader(t("budget_title"))

    col6, col7, col8 = st.columns(3)
    with col6:
        budget_str = st.text_input(t("budget"), value="5000", placeholder="e.g. 70000")
        budget = int(budget_str) if budget_str.strip().isdigit() else 5000
    with col7:
        currency_choice = st.selectbox(t("currency"), t("currencies"))
    with col8:
        style_choice = st.selectbox(t("style"), t("styles"))

    # Extract currency code only (e.g. "EGP 🇪🇬" → "EGP")
    currency_code = currency_choice.split()[0] if currency_choice else "USD"

    interests = st.multiselect(
        t("interests"),
        t("interest_opts"),
        default=t("default_interests")
    )

    special_requests = st.text_area(t("special"), placeholder=t("special_ph"), height=80)

    submitted = st.form_submit_button(t("btn"), type="primary")

# ─── Run Agent ─────────────────────────────────────────────────
if submitted:
    if not from_city:
        st.error(t("no_from"))
        st.stop()
    if not to_city:
        st.error(t("no_to"))
        st.stop()
    if not openai_key:
        st.error(t("no_key"))
        st.stop()

    # Set API keys from sidebar input
    os.environ["OPENAI_API_KEY"]          = openai_key
    os.environ["LANGCHAIN_TRACING_V2"]    = "true"
    os.environ["LANGCHAIN_PROJECT"]       = "Travel_agent"
    os.environ["LANGCHAIN_ENDPOINT"]      = "https://api.smith.langchain.com"
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    return_date = departure + timedelta(days=num_days)
    # Detect if traveler is in their home country
    nationality_str = nationality.strip() if nationality.strip() else "unknown"
    from_country = from_city.split(",")[-1].strip() if "," in from_city else from_city.strip()
    to_country = to_city.split(",")[-1].strip() if "," in to_city else to_city.strip()

    question = f"""
    I want to travel from {from_city} to {to_city}.
    EXACT DESTINATION: {to_city} — all hotels, restaurants, and activities MUST be in {to_city} only.
    Traveler nationality: {nationality_str}
    Departure: {departure.strftime('%B %d, %Y')}
    Return: {return_date.strftime('%B %d, %Y')} ({num_days} days)
    Number of travelers: {num_travelers}
    Total budget: {budget} {currency_code}
    Preferred currency: {currency_code}
    Travel style: {style_choice}
    Interests: {', '.join(interests) if interests else 'general sightseeing'}
    {f'Special requests: {special_requests}' if special_requests else ''}

    LOCATION RULE — CRITICAL:
    - Every hotel, restaurant, activity, and place MUST be physically located in {to_city}.
    - NEVER recommend anything from a different city or area.
    - If you cannot verify a place is in {to_city}, do not mention it.

    BUDGET RULE — HARD MAXIMUM:
    - Total budget: {budget} {currency_code} — ABSOLUTE MAXIMUM, never exceed it.
    - Per day budget: {budget // num_days if num_days > 0 else budget} {currency_code}
    - Hotel meal plans: All-Inclusive (75% of budget) / Full Board (65%) / Half Board (55%) / B&B (50%)
    - Always check meal plan before adding food budget — All-Inclusive and Full Board need no extra food budget.
    - Final total must be LESS than {budget} {currency_code}. Leave 5% buffer.

    CURRENCY RULE:
    - Use {currency_code} for ALL prices. Never use USD unless {currency_code} is USD.
    - Currency symbols: EGP=ج.م, USD=$, EUR=€, GBP=£, SAR=ر.س, AED=د.إ, JPY=¥, TRY=₺

    VISA INTELLIGENCE:
    - If traveler nationality matches destination country, skip visa section entirely.
    - Otherwise research visa for {nationality_str} passport holder traveling to {to_city}.

    Please find transport, hotels, weather, activities, places, and budget breakdown.
    Respond in {'Arabic' if is_ar else 'English'}.
    """

    from travel_agent import app, GraphState

    initial_state: GraphState = {
        "question": question,
        "plan": None,
        "flight_notes": [], "hotel_notes": [], "visa_notes": [],
        "weather_notes": [], "activities_notes": [], "places_notes": [],
        "budget_notes": [], "draft": None, "review": None,
        "iteration": 0, "max_iterations": 2,
    }

    st.markdown("---")
    if is_ar:
        st.markdown(
            f'<div class="rtl"><h3>{t("planning")} {to_city} — {num_days} {t("days_label")}</h3></div>',
            unsafe_allow_html=True)
    else:
        st.subheader(f"{t('planning')} {to_city} — {num_days} {t('days_label')}")

    progress_bar = st.progress(0)
    status_text  = st.empty()

    import threading, time
    steps = t("steps")
    pcts  = [10, 25, 37, 49, 61, 73, 83, 90, 95, 98]

    def update_progress():
        for pct, msg in zip(pcts, steps):
            progress_bar.progress(pct)
            status_text.info(msg)
            time.sleep(8)

    threading.Thread(target=update_progress, daemon=True).start()

    try:
        result = app.invoke(initial_state)
        progress_bar.progress(100)
        status_text.success(t("ready"))

        # ── Summary metrics ──────────────────────────────────────
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric(t("dest_label"),      to_city)
        col_b.metric(t("duration_label"),  f"{num_days} {t('days_label')}")
        col_c.metric(t("travelers_label"), num_travelers)
        col_d.metric(t("budget_label"),    f"{budget:,} {currency_code}")

        # ── Internal logs (terminal only, never shown to user) ───
        if result.get("review"):
            score = result["review"]["score"]
            print(f"[INTERNAL] Quality score: {score}/100")
            if result["review"].get("budget_check", {}).get("is_over_budget"):
                print("[INTERNAL] WARNING: Plan may exceed budget")
            if not result["review"].get("visa_check", {}).get("visa_info_present", True):
                print("[INTERNAL] WARNING: Visa info missing")

        # ── Result tabs ──────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([t("tab1"), t("tab2"), t("tab3"), t("tab4")])

        with tab1:
            if is_ar:
                st.markdown(f'<div class="rtl">{result.get("draft","")}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(result.get("draft", ""))
            st.download_button(
                t("download"),
                result.get("draft", ""),
                file_name=f"trip_{to_city}_{departure}.md",
                mime="text/markdown",
            )

        with tab2:
            # ── Real booking links (built from trip data, always work) ──
            from_clean = from_city.replace(" ", "+").replace(",", "")
            to_clean = to_city.replace(" ", "+").replace(",", "")
            dep_str = departure.strftime("%Y-%m-%d")
            ret_str = (departure + timedelta(days=num_days)).strftime("%Y-%m-%d")

            st.subheader("🔗 " + ("روابط الحجز المباشرة" if is_ar else "Direct Booking Links"))
            col_f, col_h = st.columns(2)

            with col_f:
                st.markdown("**" + ("✈️ بحث عن رحلات طيران" if is_ar else "✈️ Search Flights") + "**")
                google_flights = f"https://www.google.com/travel/flights?q=flights+from+{from_clean}+to+{to_clean}"
                skyscanner = f"https://www.skyscanner.net/transport/flights/{from_clean}/{to_clean}/{departure.strftime('%y%m%d')}/"
                kayak = f"https://www.kayak.com/flights/{from_clean}-{to_clean}/{dep_str}/{ret_str}"
                st.markdown(f"[🔍 Google Flights]({google_flights})")
                st.markdown(f"[🔍 Skyscanner]({skyscanner})")
                st.markdown(f"[🔍 Kayak]({kayak})")
                st.markdown(f"[🔍 EgyptAir](https://www.egyptair.com)")

            with col_h:
                st.markdown("**" + ("🏨 بحث عن فنادق" if is_ar else "🏨 Search Hotels") + "**")
                booking = f"https://www.booking.com/searchresults.html?ss={to_clean}&checkin={dep_str}&checkout={ret_str}&group_adults={num_travelers}"
                airbnb = f"https://www.airbnb.com/s/{to_clean}/homes?checkin={dep_str}&checkout={ret_str}&adults={num_travelers}"
                hotels_com = f"https://www.hotels.com/search.do?q-destination={to_clean}&q-check-in={dep_str}&q-check-out={ret_str}"
                st.markdown(f"[🔍 Booking.com]({booking})")
                st.markdown(f"[🔍 Airbnb]({airbnb})")
                st.markdown(f"[🔍 Hotels.com]({hotels_com})")

            st.markdown("---")

            st.subheader(t("flights"))
            for n in result.get("flight_notes", []):
                if n.strip(): st.markdown(f"- {n}")

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(t("visa"))
                for n in result.get("visa_notes", []):
                    if n.strip(): st.markdown(f"- {n}")
            with c2:
                st.subheader(t("weather"))
                for n in result.get("weather_notes", []):
                    if n.strip(): st.markdown(f"- {n}")

        with tab4:
            st.subheader(t("budget_break"))
            for n in result.get("budget_notes", []):
                if n.strip(): st.markdown(f"- {n}")
            # Internal only
            if result.get("review", {}).get("budget_check", {}).get("is_over_budget"):
                print("[INTERNAL] Budget exceeded - consider revising plan")

    except Exception as e:
        progress_bar.empty()
        print(f"[INTERNAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        status_text.error(t("error"))