# ✈️ AI Travel Planner

A multi-agent AI travel planning system built with **LangGraph** and **LangChain**. Ask it where you want to go, set your budget, and it researches flights, hotels, visa requirements, weather, activities, and more — then writes you a complete itinerary.

🌐 **Live Demo:** [ai-travel-planner.mohamedmagdy.cfd](https://ai-travel-planner.mohamedmagdy.cfd)

---

## 🤖 How It Works — 11 Specialized Agents

```
Planner → Flight → Hotel → Visa → Weather → Activities → Places → Budget → Writer → Reviewer → Finalizer
```

Each agent has one job:

| Agent | What It Does |
|-------|-------------|
| **Planner** | Understands the request, routes to the right agents |
| **Flight** | Researches flight options and prices |
| **Hotel** | Finds 3 hotel options at different price points |
| **Visa** | Checks visa requirements for your nationality |
| **Weather** | Gets weather forecast for travel dates |
| **Activities** | Suggests things to do |
| **Places** | Recommends restaurants and attractions |
| **Budget** | Allocates and tracks the total budget |
| **Writer** | Writes the full itinerary |
| **Reviewer** | Reviews quality and flags issues |
| **Finalizer** | Delivers the final polished plan |

The **Reviewer** can send the plan back for revision if quality is below threshold — the system loops until it's good enough or hits the max iteration limit.

---

## 🛠️ Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Multi-agent orchestration with stateful graph
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM framework
- **[OpenAI GPT-4o-mini](https://openai.com)** — Language model (temperature=0.2)
- **[Tavily Search](https://tavily.com)** — Real-time web search for all research agents
- **[Streamlit](https://streamlit.io)** — Web UI
- **[LangSmith](https://smith.langchain.com)** — Observability and tracing
- **Python 3.12**

---

## ✨ Features

- 🌍 **Bilingual** — English and Arabic UI
- 💱 **8 Currencies** — EGP, USD, EUR, GBP, SAR, AED, JPY, TRY
- 🛂 **Smart Visa Logic** — Nationality-aware (skips visa section for domestic travel)
- 🚗 **No-Airport Detection** — Suggests ground transport for destinations without airports
- 🏨 **Hotel Meal Plan Logic** — Budget allocation changes based on meal plan type
- 🔗 **Real Booking Links** — Built from actual trip data (not AI-generated)
- 🚫 **Anti-Hallucination Rules** — All hotels and places verified via Tavily before recommending
- 📊 **LangSmith Tracing** — Full observability on every agent run

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/mohamedmagdy776/travel-agent
cd travel-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys
Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Travel_agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 4. Run the app
```bash
streamlit run travel_streamlit.py
```

---

## 📁 Project Structure

```
travel-agent/
├── travel_agent.py        # LangGraph multi-agent brain (11 agents)
├── travel_streamlit.py    # Streamlit web UI
├── requirements.txt       # Pinned dependencies
└── .env                   # API keys (not committed)
```

---

## 👤 Author

**Mohamed Magdy** — Career transition from Petroleum Engineering to AI Development.

---

## 📄 License

MIT
