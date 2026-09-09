"""
Multi-Agent Travel Planner using LangGraph
==========================================
11 Agents: Planner → Flight → Hotel → Visa → Weather → Activities → Places
           → Budget → Writer → Reviewer → Finalizer

Requirements:
    pip install langchain langchain-openai langchain-community langgraph pydantic python-dotenv tavily-python

Setup:
    - OPENAI_API_KEY
    - TAVILY_API_KEY  (free at tavily.com)
    - LANGSMITH_API_KEY  (optional)
    - LANGCHAIN_TRACING_V2=true  (optional)
    - LANGCHAIN_PROJECT=travel-agent  (optional)
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────
# 1. LLM + Search Tool
# ─────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]


# ─────────────────────────────────────────────
# 2. Structured Outputs
# ─────────────────────────────────────────────
class TravelPlan(BaseModel):
    destination: str = Field(..., description="Travel destination")
    travel_dates: str = Field(..., description="Proposed travel dates")
    num_days: int = Field(..., description="Number of travel days")
    traveler_preferences: List[str] = Field(..., description="What the traveler enjoys")
    steps: List[str] = Field(..., description="Ordered planning steps")
    key_risks: List[str] = Field(..., description="Major risks or unknowns")
    desired_output_structure: List[str] = Field(..., description="Headings for final itinerary")


class BudgetCheck(BaseModel):
    user_budget: Optional[float] = Field(None, description="Budget the user specified (USD)")
    estimated_total: Optional[float] = Field(None, description="Total estimated cost (USD)")
    is_over_budget: bool = Field(False)
    overage_amount: Optional[float] = Field(None)
    savings_suggestions: List[str] = Field(default_factory=list)


class LogisticsCheck(BaseModel):
    hotel_far_from_activities: bool = Field(False)
    unrealistic_schedule: bool = Field(False)
    missing_transit_info: bool = Field(False)
    airport_transfer_missing: bool = Field(False)
    details: List[str] = Field(default_factory=list)


class VisaCheck(BaseModel):
    visa_info_present: bool = Field(True)
    visa_info_complete: bool = Field(True)
    passport_validity_mentioned: bool = Field(False)
    details: List[str] = Field(default_factory=list)


class SafetyCheck(BaseModel):
    travel_warnings_mentioned: bool = Field(False)
    insurance_mentioned: bool = Field(False)
    emergency_contacts_included: bool = Field(False)
    details: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    budget_check: BudgetCheck = Field(default_factory=BudgetCheck)
    logistics_check: LogisticsCheck = Field(default_factory=LogisticsCheck)
    visa_check: VisaCheck = Field(default_factory=VisaCheck)
    safety_check: SafetyCheck = Field(default_factory=SafetyCheck)
    weather_appropriate: bool = Field(True, description="Activities match the weather")
    hallucination_risk: List[str] = Field(default_factory=list)
    missing_points: List[str] = Field(default_factory=list)
    score: int = Field(..., ge=0, le=100)
    fix_instructions: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# 3. Graph State
# ─────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    plan: Optional[Dict[str, Any]]
    flight_notes: List[str]
    hotel_notes: List[str]
    visa_notes: List[str]
    weather_notes: List[str]
    activities_notes: List[str]
    places_notes: List[str]
    budget_notes: List[str]
    draft: Optional[str]
    review: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int


# ─────────────────────────────────────────────
# 4. System Prompts
# ─────────────────────────────────────────────

PLANNER_SYSTEM = """You are the Planner agent for a travel planning system.
Given a user's travel request:
1. Identify destination, dates, number of days, and traveler preferences.
2. Break planning into ordered steps.
3. Identify key risks (visa, weather, budget, safety).
4. Define output headings for the final itinerary.

IMPORTANT — Transport intelligence:
- If the destination has NO commercial airport (e.g. Marsa Matruh, Siwa, Dahab, Nuweiba),
  add to key_risks: 'No direct flights — ground transport required (bus/car)'
- Suggest the nearest hub airport and ground transport instead of flights.
- Egyptian no-airport destinations: Marsa Matruh (bus from Cairo 5-6hrs),
  Siwa (bus 8hrs), Dahab (fly to Sharm + drive 1.5hrs)
- Never recommend flights to destinations with no airport.

Return valid JSON matching the TravelPlan schema."""

FLIGHT_SYSTEM = """You are the Flight agent. You have access to a web search tool called Tavily.
Rules:
- Search for real flight options for the user's route and dates.
- Find best airlines, price ranges (economy/business), flight duration, layovers.
- Include booking tips and best time to buy.
- DO NOT include any booking links — links will be added automatically by the system.
- If destination has no airport, suggest ground transport options with cost and duration instead.
- Use Tavily when the question requires current, recent info.
- Don't use Tavily when web search is unnecessary.
- After searching, formulate bullet-point notes with airlines, prices, duration, and tips."""

HOTEL_SYSTEM = """You are the Hotel agent. You have access to a web search tool called Tavily.

CRITICAL RULES:
1. LOCATION: Search ONLY in the EXACT destination. Never show a hotel from a different city.

2. HOTEL TYPE INTELLIGENCE:
   Hotels always include some meals. Types and budget allocation:
   - ALL-INCLUSIVE: covers all meals + drinks + most activities.
     Hotel can take up to 75% of total budget. Outside spending = minimal.
   - FULL BOARD: covers breakfast + lunch + dinner.
     Hotel can take up to 65% of total budget. Outside spending = activities + transport only.
   - HALF BOARD: covers breakfast + dinner.
     Hotel can take up to 55% of total budget. Lunch outside = add 10% for food.
   - BED & BREAKFAST: covers breakfast only.
     Hotel can take up to 50% of total budget. Lunch + dinner outside = add 20% for food.

3. BUDGET: Max per night = total budget × hotel_percentage / number of nights
   - Always calculate max per night based on hotel type above.
   - NEVER recommend a hotel above this calculated maximum.
   - Always mention meal plan type for each hotel.

4. Always provide EXACTLY 3 options at DIFFERENT price points:
   - Option 1: best value (lower price, good quality)
   - Option 2: mid-range
   - Option 3: premium (if within budget)
   - ALL three must stay within the budget constraint.

For each hotel:
  * Exact name (verified in destination)
  * Star rating
  * Price per night in preferred currency
  * What is INCLUDED (all-inclusive / half-board / room only / breakfast)
  * Location within destination
  * What makes it special (beach access, pool, spa, etc.)
  * Rating if available
  * Best for (families, couples, etc.)

- Add RECOMMENDATION: best choice for this traveler considering budget AND what is included.
- Use Tavily to search and verify real hotels in the destination.
- Show budget calculation before the 3 options."""

VISA_SYSTEM = """You are the Visa agent. You have access to a web search tool called Tavily.

FIRST — Read the traveler nationality and destination carefully:
- If the traveler is traveling WITHIN their own country (e.g. Egyptian going to Hurghada, Egypt),
  respond with exactly: "No visa required — traveler is within their home country." and STOP.
  Do not search for anything. Do not write anything else.
- If destination is in the same country as nationality, same rule applies.

If visa research IS needed:
- Search specifically for visa requirements for THAT nationality passport traveling to THAT destination.
- Never give generic visa info — always specify the passport nationality.
- Cover: visa type, required documents, fees, processing time, e-visa options.
- Mention passport validity requirements (usually 6 months beyond travel dates).
- Include visa-on-arrival options if available for that nationality.
- Warn about common rejection reasons specific to that nationality.
- Use Tavily for current, accurate info.
- After searching, formulate specific bullet-point notes."""

WEATHER_SYSTEM = """You are the Weather agent. You have access to a web search tool called Tavily.
Rules:
- Search for weather conditions at the destination during the travel dates.
- Cover: temperature range, rainfall/snow, humidity.
- Recommend what to pack (clothing, umbrella, sunscreen, etc.).
- Warn about extreme weather (typhoons, monsoons, heatwaves).
- Suggest best/worst months if dates are flexible.
- Use Tavily for current weather forecasts and seasonal patterns.
- Don't use Tavily when web search is unnecessary.
- After searching, use the retrieved information to formulate bullet-point notes."""

ACTIVITIES_SYSTEM = """You are the Activities agent. You have access to a web search tool called Tavily.
Rules:
- Search for SPECIFIC named activities at the destination.
- For each activity include: exact name, location, price, duration, rating (if available), why it is special.
- Examples of good output: 
  "Snorkeling at Elphinstone Reef — rated 4.9/5, one of top 10 dive sites in the world, depth 50m, 45 min by boat from port, costs 0-50"
  "Quad biking in the desert with Marsa Alam Adventures — 2 hours, 5/person, pickup from hotel included"
- NEVER write vague things like "enjoy water sports" or "explore the area"
- Always name the specific operator, location, or site
- Use Tavily to search for current, real activities with names and prices.
- After searching, formulate specific bullet-point notes with names and details."""

PLACES_SYSTEM = """You are the Places agent. You have access to a web search tool called Tavily.
Rules:
- Search for SPECIFIC named places at the destination.
- For each place include: exact name, what it is, why it is special, rating, price/entry fee, opening hours, location.
- Examples of good output:
  "Abu Dabbab Beach — famous for sea turtles and dugongs, one of few places in Egypt where you can swim with dugongs, free entry, 30 min north of Marsa Alam"
  "Sharm El Luli — secluded pink-sand beach, rated 4.9/5 on Google, accessible only by boat (20 min, 5), no facilities so bring food"
  "Hamata Mangroves — unique mangrove forest, great for kayaking, 1hr south of Marsa Alam, entry free"
- For restaurants: name, cuisine type, price range, must-order dish, rating
- NEVER say 'visit local restaurants' or 'explore the area'  
- Use Tavily to search for real places with names and details.
- After searching, formulate specific bullet-point notes."""

BUDGET_SYSTEM = """You are the Budget agent. You have access to a web search tool called Tavily.

BUDGET IS A HARD MAXIMUM — MOST IMPORTANT RULE:
- The total budget stated by the user is the ABSOLUTE MAXIMUM they want to spend.
- NEVER recommend a plan that costs more than the budget.
- Always leave a small buffer (10-15%) under the budget for unexpected expenses.

BUDGET MATH — always calculate:
1. Per-day budget = total budget / number of days
2. Per-person-per-day = per-day budget / number of travelers

SMART ALLOCATION based on hotel meal plan:
- ALL-INCLUSIVE: Hotel 75% | Outside activities 15% | Transport 5% | Misc 5%
  (No food budget needed — all included)
- FULL BOARD: Hotel 65% | Outside activities 20% | Transport 10% | Misc 5%
  (No food budget needed — all meals included)
- HALF BOARD: Hotel 55% | Lunch outside 10% | Activities 20% | Transport 10% | Misc 5%
  (Only lunch needs budget)
- BED & BREAKFAST: Hotel 50% | Lunch+Dinner 20% | Activities 15% | Transport 10% | Misc 5%

RULES:
- Total must NEVER exceed the budget. Leave 5% buffer.
- Never add food budget if hotel is all-inclusive or full board.
- Show final breakdown table clearly.

- Use preferred currency for ALL prices.
- Use Tavily for current prices.
- After searching, formulate specific bullet-point notes with the budget math shown clearly."""

WRITER_SYSTEM = """You are the Writer agent for a travel planning system.
Write a detailed, specific, useful travel plan.

STRICT RULES — never do these:
- NEVER write vague phrases like "explore the city", "enjoy the nightlife", "discover local culture"
- NEVER recommend something without a specific name
- NEVER say "visit a local restaurant" — always name the actual restaurant
- NEVER say "check out attractions" — always name the specific attraction

ALWAYS do these:
- Name EVERY place, restaurant, activity specifically (e.g. "White Island (Gezira Baida) — rated 4.8/5, known for white sand")
- Include why each place is recommended (what makes it special, what it's known for)
- Include practical info: opening hours, entry fees, how to get there, tips
- For each day: morning/afternoon/evening with NAMED places only
- Include ratings when available (from research notes)
- Include distances and travel times between places
- Packing list based on actual weather research
- Emergency contacts: nearest hospital, police, embassy

FOR HOTELS — present the 3 hotels ONCE ONLY in a comparison table. Never mention hotels again anywhere else:
| الفندق | النجوم | السعر/ليلة | المميزات | التقييم |
|--------|--------|------------|----------|---------|
| فندق 1 | ⭐⭐⭐⭐⭐ | X ج.م | كذا | 9.2 |
| فندق 2 | ⭐⭐⭐⭐⭐ | X ج.م | كذا | 8.8 |
| فندق 3 | ⭐⭐⭐⭐ | X ج.م | كذا | 8.5 |
Then write: "🏆 توصية الـ AI: [اسم الفندق] — لأن [السبب]"
IMPORTANT: Do NOT repeat or mention hotels anywhere else in the plan.

FOR FLIGHTS/TRANSPORT — present options in a table ONCE ONLY (NO links):
| وسيلة النقل | المدة | التكلفة | الملاحظات |
|------------|-------|---------|-----------|
| option 1 | Xhr | X ج.م | details |
IMPORTANT: Do NOT repeat transport options anywhere else.

BUDGET ALIGNMENT — CRITICAL:
- Read the traveler's total budget carefully.
- If budget is high, recommend premium experiences throughout the plan.
- Make sure the day-by-day plan actually uses the budget well.
- Show at the end: total estimated spend vs budget, and what to do with remaining budget.

Use ALL research notes: flights/transport, hotels, visa, weather, activities, places, budget.
Include quick-reference summary table at top.
If a review exists, incorporate the fixes."""

REVIEWER_SYSTEM = """You are the Travel Plan Reviewer — a domain-specific validator.
You validate the LOGIC and COMPLETENESS of the travel plan.

## Budget validation
- Extract user's budget vs estimated total. Flag if over budget.
- Suggest specific savings.

## Logistics validation
- Hotel far from activities? Unrealistic daily schedule (max 3-4 major)?
- Missing transit info? No airport transfer plan?

## Visa validation
- Complete visa info? Passport validity? Common pitfalls?

## Weather validation
- Activities appropriate for weather? Packing list matches weather?

## Safety validation
- Travel warnings? Insurance? Emergency contacts?

## Scoring: Start at 100, deduct:
- Budget exceeded: -20 | Missing visa: -15 | Unrealistic schedule: -10
- No transit info: -10 | No airport transfer: -5 | No safety info: -5
- Weather mismatch: -5 | Each hallucination risk: -5

Return JSON matching the ReviewResult schema."""

FINALIZER_SYSTEM = """You are the Finalizer agent.
Produce the ultimate polished travel itinerary from all research and drafts.

Your output must include:
1. Clean summary table (destination, dates, budget, visa status)
2. Day-by-day itinerary with times, places, costs
3. Practical info (transport, money, language, safety)
4. Packing checklist (based on weather)
5. Emergency info (embassy, hospital, police numbers)

If a review exists, incorporate ALL fixes.
Make it ready to print and follow.

IMPORTANT: Do NOT include any confidence score, quality score, rating score,
or any numerical evaluation in your output. Never write phrases like
"درجة الثقة", "Confidence Score", "Quality Score", or any score number."""


# ─────────────────────────────────────────────
# 5. Helper
# ─────────────────────────────────────────────

def _make_research_agent(system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def _parse_notes(text: str) -> List[str]:
    return [line.strip("- ").strip() for line in text.split("\n") if line.strip()]


# ─────────────────────────────────────────────
# 6. Agent Nodes
# ─────────────────────────────────────────────

def planner_node(state: GraphState) -> GraphState:
    structured_planner = llm.with_structured_output(TravelPlan)
    plan_obj = structured_planner.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=state["question"]),
    ])
    state["plan"] = plan_obj.model_dump()
    return state


def flight_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(FLIGHT_SYSTEM)
    result = agent.invoke({
        "input": f"Find flights for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}"
    })
    state["flight_notes"] = _parse_notes(result["output"])
    return state


def hotel_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(HOTEL_SYSTEM)
    result = agent.invoke({
        "input": f"Find hotels for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}\nDays: {state['plan']['num_days']}"
    })
    state["hotel_notes"] = _parse_notes(result["output"])
    return state


def visa_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(VISA_SYSTEM)
    result = agent.invoke({
        "input": f"Find visa requirements for: {state['question']}\nDestination: {state['plan']['destination']}"
    })
    state["visa_notes"] = _parse_notes(result["output"])
    return state


def weather_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(WEATHER_SYSTEM)
    result = agent.invoke({
        "input": f"Find weather for {state['plan']['destination']} during {state['plan']['travel_dates']}"
    })
    state["weather_notes"] = _parse_notes(result["output"])
    return state


def activities_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(ACTIVITIES_SYSTEM)
    result = agent.invoke({
        "input": f"Find activities and experiences for: {state['question']}\nDestination: {state['plan']['destination']}\nPreferences: {state['plan'].get('traveler_preferences', [])}"
    })
    state["activities_notes"] = _parse_notes(result["output"])
    return state


def places_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(PLACES_SYSTEM)
    result = agent.invoke({
        "input": f"Find best places, landmarks, restaurants for: {state['question']}\nDestination: {state['plan']['destination']}\nPreferences: {state['plan'].get('traveler_preferences', [])}"
    })
    state["places_notes"] = _parse_notes(result["output"])
    return state


def budget_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(BUDGET_SYSTEM)
    result = agent.invoke({
        "input": f"Calculate travel budget for: {state['question']}\nDestination: {state['plan']['destination']}\nDays: {state['plan']['num_days']}"
    })
    state["budget_notes"] = _parse_notes(result["output"])
    return state


def writer_node(state: GraphState) -> GraphState:
    headings = state["plan"].get("desired_output_structure", [])
    review_text = ""
    if state.get("review"):
        review_text = f"\n\nPrevious review to address:\n{json.dumps(state['review'], indent=2)}"

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

Plan: {json.dumps(state['plan'], indent=2)}
Output headings: {headings}

Flight Research:
{chr(10).join('- ' + n for n in state.get('flight_notes', []))}

Hotel Research:
{chr(10).join('- ' + n for n in state.get('hotel_notes', []))}

Visa Research:
{chr(10).join('- ' + n for n in state.get('visa_notes', []))}

Weather Research:
{chr(10).join('- ' + n for n in state.get('weather_notes', []))}

Activities Research:
{chr(10).join('- ' + n for n in state.get('activities_notes', []))}

Places Research:
{chr(10).join('- ' + n for n in state.get('places_notes', []))}

Budget Research:
{chr(10).join('- ' + n for n in state.get('budget_notes', []))}
{review_text}
"""),
    ]).content
    state["draft"] = resp
    return state


def reviewer_node(state: GraphState) -> GraphState:
    structured_reviewer = llm.with_structured_output(ReviewResult)
    review_obj = structured_reviewer.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"""User's original request:
{state['question']}

Draft to validate:
{state['draft']}

--- Raw research for cross-reference ---
Flights: {chr(10).join('- ' + n for n in state.get('flight_notes', []))}
Hotels: {chr(10).join('- ' + n for n in state.get('hotel_notes', []))}
Visa: {chr(10).join('- ' + n for n in state.get('visa_notes', []))}
Weather: {chr(10).join('- ' + n for n in state.get('weather_notes', []))}
Activities: {chr(10).join('- ' + n for n in state.get('activities_notes', []))}
Places: {chr(10).join('- ' + n for n in state.get('places_notes', []))}
Budget: {chr(10).join('- ' + n for n in state.get('budget_notes', []))}
"""),
    ])
    state["review"] = review_obj.model_dump()
    state["iteration"] += 1
    return state


def finalizer_node(state: GraphState) -> GraphState:
    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

Plan: {json.dumps(state['plan'], indent=2)}

Flights: {chr(10).join('- ' + n for n in state.get('flight_notes', []))}
Hotels: {chr(10).join('- ' + n for n in state.get('hotel_notes', []))}
Visa: {chr(10).join('- ' + n for n in state.get('visa_notes', []))}
Weather: {chr(10).join('- ' + n for n in state.get('weather_notes', []))}
Activities: {chr(10).join('- ' + n for n in state.get('activities_notes', []))}
Places: {chr(10).join('- ' + n for n in state.get('places_notes', []))}
Budget: {chr(10).join('- ' + n for n in state.get('budget_notes', []))}

Current Draft: {state.get('draft', '')}
Review: {json.dumps(state.get('review'), indent=2) if state.get('review') else 'None'}
"""),
    ]).content
    state["draft"] = resp
    return state


# ─────────────────────────────────────────────
# 7. Conditional Edge
# ─────────────────────────────────────────────

def should_revise(state: GraphState) -> Literal["revise", "finalize"]:
    review = state["review"]
    score = review["score"]

    if state["iteration"] >= state["max_iterations"]:
        return "finalize"

    budget = review.get("budget_check", {})
    visa = review.get("visa_check", {})
    has_critical = (
        budget.get("is_over_budget", False)
        or not visa.get("visa_info_present", True)
    )

    if score < 80 or has_critical:
        return "revise"

    return "finalize"


# ─────────────────────────────────────────────
# 8. Build the Graph
# ─────────────────────────────────────────────

workflow = StateGraph(GraphState)

workflow.add_node("planner", planner_node)
workflow.add_node("flight", flight_node)
workflow.add_node("hotel", hotel_node)
workflow.add_node("visa", visa_node)
workflow.add_node("weather", weather_node)
workflow.add_node("activities", activities_node)
workflow.add_node("places", places_node)
workflow.add_node("budget", budget_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("finalizer", finalizer_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "flight")
workflow.add_edge("flight", "hotel")
workflow.add_edge("hotel", "visa")
workflow.add_edge("visa", "weather")
workflow.add_edge("weather", "activities")
workflow.add_edge("activities", "places")
workflow.add_edge("places", "budget")
workflow.add_edge("budget", "writer")
workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    should_revise,
    {
        "revise": "writer",
        "finalize": "finalizer",
    },
)

workflow.add_edge("finalizer", END)

app = workflow.compile()

# ─────────────────────────────────────────────
# 9. Visualization
# ─────────────────────────────────────────────
try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass

# ─────────────────────────────────────────────
# 10. Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    question = input("🌍 Enter your travel request: ") or (
        "I want to travel from Cairo, Egypt to Tokyo, Japan for 10 days "
        "in December 2026. Budget is around $3000. I love food, temples, "
        "and anime culture. I need visa info too."
    )

    initial_state: GraphState = {
        "question": question,
        "plan": None,
        "flight_notes": [],
        "hotel_notes": [],
        "visa_notes": [],
        "weather_notes": [],
        "activities_notes": [],
        "places_notes": [],
        "budget_notes": [],
        "draft": None,
        "review": None,
        "iteration": 0,
        "max_iterations": 2,
    }

    print("\n⏳ Running multi-agent travel planner...\n")
    result = app.invoke(initial_state)

    print("=" * 60)
    print("✈️  FINAL TRAVEL ITINERARY")
    print("=" * 60)
    print(result["draft"])

    if result.get("review"):
        print(f"\n📊 Review score: {result['review']['score']}/100")