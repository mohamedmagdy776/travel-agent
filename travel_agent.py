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
from langgraph.prebuilt import create_react_agent
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
    num_days: int = Field(..., description="Number of travel days — must match EXACTLY what the user requested")
    travel_style: str = Field("mid-range", description="Travel style: luxury, mid-range, or budget")
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

CRITICAL — num_days:
- The user specifies EXACTLY how many days they want. Extract this number precisely.
- num_days must match the user's request EXACTLY. If they say "3 days", num_days = 3.
- Never add or subtract days from what the user requested.

CRITICAL — travel_style:
- Extract the travel style from the user's request (luxury, mid-range, budget).
- If user says "فاخر" or "luxury", set travel_style = "luxury".
- If user says "متوسط" or "mid-range", set travel_style = "mid-range".
- If user says "اقتصادي" or "budget", set travel_style = "budget".
- travel_style MUST influence ALL downstream recommendations.

IMPORTANT — Transport intelligence:
- If the destination has NO commercial airport (e.g. Marsa Matruh, Siwa, Dahab, Nuweiba),
  add to key_risks: 'No direct flights — ground transport required (bus/car)'
- Suggest the nearest hub airport and ground transport instead of flights.
- Egyptian no-airport destinations: Marsa Matruh (bus from Cairo 5-6hrs),
  Siwa (bus 8hrs), Dahab (fly to Sharm + drive 1.5hrs)
- Never recommend flights to destinations with no airport.
- For LUXURY travel style: always prefer the fastest/most comfortable transport option.
  If bus takes 8+ hours but flights exist to a nearby airport, recommend flight + private transfer.

Return valid JSON matching the TravelPlan schema."""

FLIGHT_SYSTEM = """You are the Flight/Transport agent. You have access to a web search tool called Tavily.

TRAVEL STYLE IS CRITICAL:
- LUXURY: Recommend flights (business class if available), private transfers, premium transport ONLY.
  NEVER recommend a public bus or shared minibus for luxury travelers.
  If no direct flights exist, recommend: flight to nearest airport + private car/taxi transfer.
  Example: Dahab luxury = fly to Sharm El Sheikh + private transfer (1.5hrs, ~500-800 EGP).
- MID-RANGE: Recommend economy flights, mix of private and shared transport.
- BUDGET: Recommend cheapest options — buses, shared transport, budget airlines.

Rules:
- Search for real transport options for the user's route and dates.
- Find best airlines, price ranges, flight duration, layovers.
- Include booking tips and best time to buy.
- DO NOT include any booking links — links will be added automatically by the system.
- If destination has no airport, suggest the BEST transport option for the travel style (see above).
- Use Tavily when the question requires current, recent info.
- Don't use Tavily when web search is unnecessary.

OUTPUT RULES — VERY IMPORTANT:
- ONLY include transport/flight information: airlines, routes, prices, duration, tips.
- DO NOT calculate total trip budget or budget breakdown — that is the Budget agent's job.
- DO NOT include hotel costs, activity costs, or food costs.
- ONLY write about how to GET THERE and GET BACK.
- After searching, formulate bullet-point notes with transport options, prices, and duration."""

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

CRITICAL — USE EXACT NUM_DAYS:
- The number of days is provided in the query. Use EXACTLY that number.
- If the user says 1 day, calculate budget for 1 day ONLY.
- If the user says 3 days, calculate budget for 3 days ONLY.
- NEVER calculate for more or fewer days than specified.

BUDGET MATH — always calculate:
1. Per-day budget = total budget / number of days (use EXACT num_days from query)
2. Per-person-per-day = per-day budget / number of travelers

TRAVEL STYLE IMPACT:
- LUXURY: allocate more to premium experiences, 5-star hotels, private transport.
- MID-RANGE: balanced allocation.
- BUDGET: maximize savings, cheapest options.

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
- YOUR OUTPUT IS THE COMPLETE BUDGET SECTION — include total costs by category,
  daily breakdown, comparison to user's budget, and remaining buffer.

- Use preferred currency for ALL prices.
- Use Tavily for current prices.
- After searching, formulate specific bullet-point notes with the budget math shown clearly."""

WRITER_SYSTEM = """You are the Writer agent for a travel planning system.
Write a detailed, specific, useful travel plan.

═══════════════════════════════════════════════════
DAY COUNT — THE MOST CRITICAL RULE:
═══════════════════════════════════════════════════
- The plan JSON contains "num_days". This is the EXACT number of days the user wants.
- Your day-by-day itinerary MUST have EXACTLY num_days days. No more, no less.
- If num_days = 1, write ONLY "Day 1". NEVER add Day 2.
- If num_days = 3, write Day 1, Day 2, Day 3. NEVER add Day 4.
- If num_days = 7, write Day 1 through Day 7. NEVER add Day 8.
- VIOLATING THIS RULE IS THE WORST POSSIBLE ERROR.

═══════════════════════════════════════════════════
TRAVEL STYLE — MUST INFLUENCE EVERYTHING:
═══════════════════════════════════════════════════
- The plan JSON contains "travel_style".
- LUXURY: recommend premium restaurants, 5-star experiences, private tours, fine dining.
  Never recommend street food stalls or cheap eateries as main options.
- MID-RANGE: mix of nice restaurants and local gems.
- BUDGET: focus on value, street food, free activities.

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

═══════════════════════════════════════════════════
BUDGET SECTION — PLACEMENT RULES:
═══════════════════════════════════════════════════
- Put the FULL budget breakdown (total costs, category breakdown, daily breakdown) at the END of the plan
  in a clearly labeled section: "💰 الميزانية" or "💰 Budget".
- DO NOT put budget totals or breakdowns inside the transport/flights section.
- The transport section should ONLY show transport option prices, NOT total trip budget.
- Budget section should show: transport cost + hotel cost + activities cost + food cost = TOTAL
  and compare to user's budget.

Use ALL research notes: flights/transport, hotels, visa, weather, activities, places, budget.
Include quick-reference summary table at top.
If a review exists, incorporate the fixes."""

REVIEWER_SYSTEM = """You are the Travel Plan Reviewer — a domain-specific validator.
You validate the LOGIC and COMPLETENESS of the travel plan.

## DAY COUNT validation — MOST CRITICAL
- The user requested a specific num_days. Count the days in the itinerary.
- If the itinerary has MORE days than num_days, this is a CRITICAL error (-25 points).
- If the itinerary has FEWER days than num_days, this is also an error (-15 points).
- Add to fix_instructions: "Day count mismatch: user requested X days but itinerary has Y days. Fix to exactly X days."

## Travel Style validation
- Check if recommendations match the travel style (luxury/mid-range/budget).
- If travel_style is "luxury" but plan recommends buses or budget hotels: -15 points.
- Add to fix_instructions: "Travel style mismatch: user chose luxury but plan recommends [budget option]. Replace with premium alternative."

## Budget validation
- Extract user's budget vs estimated total. Flag if over budget.
- Suggest specific savings.
- Check that the full budget breakdown is NOT inside the transport section.

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
- Day count mismatch: -25 | Travel style mismatch: -15
- Budget exceeded: -20 | Missing visa: -15 | Unrealistic schedule: -10
- No transit info: -10 | No airport transfer: -5 | No safety info: -5
- Weather mismatch: -5 | Each hallucination risk: -5
- Budget in wrong section: -10

Return JSON matching the ReviewResult schema."""

FINALIZER_SYSTEM = """You are the Finalizer agent.
Produce the ultimate polished travel itinerary from all research and drafts.

═══════════════════════════════════════════════════
DAY COUNT — ABSOLUTE RULE:
═══════════════════════════════════════════════════
- The plan JSON has "num_days". Your itinerary MUST have EXACTLY that many days.
- If num_days = 1, write ONLY Day 1. NEVER write Day 2.
- If num_days = 5, write Day 1 through Day 5. NEVER write Day 6.
- Count the days in your output before finishing. If the count doesn't match num_days, FIX IT.

═══════════════════════════════════════════════════
TRAVEL STYLE — MUST BE REFLECTED:
═══════════════════════════════════════════════════
- The plan JSON has "travel_style". Every recommendation must match it.
- LUXURY: premium everything — 5-star hotels, flights/private transfers, fine dining.
- MID-RANGE: good quality balance.
- BUDGET: cheapest practical options.
- If the draft recommends a bus for a luxury traveler, REPLACE it with flights/private transfer.

Your output must include:
1. Clean summary table (destination, dates, budget, visa status)
2. Day-by-day itinerary with times, places, costs (EXACTLY num_days days)
3. Practical info (transport, money, language, safety)
4. Packing checklist (based on weather)
5. Emergency info (embassy, hospital, police numbers)
6. Budget breakdown at the END (total by category vs user budget)

If a review exists, incorporate ALL fixes.
Make it ready to print and follow.

BUDGET PLACEMENT:
- The complete budget summary goes at the END under "💰 الميزانية" or "💰 Budget".
- Transport section shows only transport options and prices.
- NEVER put the full budget breakdown inside the transport/flights section.

IMPORTANT: Do NOT include any confidence score, quality score, rating score,
or any numerical evaluation in your output. Never write phrases like
"درجة الثقة", "Confidence Score", "Quality Score", or any score number."""


# ─────────────────────────────────────────────
# 5. Helper
# ─────────────────────────────────────────────

def _make_research_agent(system_prompt: str):
    """Return (agent, system_prompt) tuple for later invocation."""
    return create_react_agent(llm, tools), system_prompt


def _invoke_agent(agent_tuple, query: str) -> str:
    """Invoke a research agent and extract the final text response."""
    agent, system_prompt = agent_tuple
    result = agent.invoke({"messages": [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]})
    return result["messages"][-1].content


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
    travel_style = state['plan'].get('travel_style', 'mid-range')
    output = _invoke_agent(agent,
        f"Find transport for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}\nTravel style: {travel_style}\nIMPORTANT: Travel style is '{travel_style}' — recommend transport that matches this style."
    )
    state["flight_notes"] = _parse_notes(output)
    return state


def hotel_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(HOTEL_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')
    output = _invoke_agent(agent,
        f"Find hotels for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}\nDays: {state['plan']['num_days']}\nTravel style: {travel_style}"
    )
    state["hotel_notes"] = _parse_notes(output)
    return state


def visa_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(VISA_SYSTEM)
    output = _invoke_agent(agent,
        f"Find visa requirements for: {state['question']}\nDestination: {state['plan']['destination']}"
    )
    state["visa_notes"] = _parse_notes(output)
    return state


def weather_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(WEATHER_SYSTEM)
    output = _invoke_agent(agent,
        f"Find weather for {state['plan']['destination']} during {state['plan']['travel_dates']}"
    )
    state["weather_notes"] = _parse_notes(output)
    return state


def activities_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(ACTIVITIES_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')
    output = _invoke_agent(agent,
        f"Find activities and experiences for: {state['question']}\nDestination: {state['plan']['destination']}\nPreferences: {state['plan'].get('traveler_preferences', [])}\nTravel style: {travel_style}"
    )
    state["activities_notes"] = _parse_notes(output)
    return state


def places_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(PLACES_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')
    output = _invoke_agent(agent,
        f"Find best places, landmarks, restaurants for: {state['question']}\nDestination: {state['plan']['destination']}\nPreferences: {state['plan'].get('traveler_preferences', [])}\nTravel style: {travel_style}"
    )
    state["places_notes"] = _parse_notes(output)
    return state


def budget_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(BUDGET_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')
    num_days = state['plan']['num_days']
    output = _invoke_agent(agent,
        f"Calculate travel budget for: {state['question']}\nDestination: {state['plan']['destination']}\nEXACT number of days: {num_days} (calculate for {num_days} days ONLY, not more)\nTravel style: {travel_style}"
    )
    state["budget_notes"] = _parse_notes(output)
    return state


def writer_node(state: GraphState) -> GraphState:
    headings = state["plan"].get("desired_output_structure", [])
    review_text = ""
    if state.get("review"):
        review_text = f"\n\nPrevious review to address:\n{json.dumps(state['review'], indent=2)}"

    num_days = state['plan']['num_days']
    travel_style = state['plan'].get('travel_style', 'mid-range')

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

══════ CRITICAL CONSTRAINTS ══════
NUMBER OF DAYS: {num_days} — write EXACTLY {num_days} day(s) in the itinerary. Not {num_days + 1}, not {num_days - 1}.
TRAVEL STYLE: {travel_style} — ALL recommendations must match this style.
══════════════════════════════════

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
    num_days = state['plan']['num_days']
    travel_style = state['plan'].get('travel_style', 'mid-range')
    structured_reviewer = llm.with_structured_output(ReviewResult)
    review_obj = structured_reviewer.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"""User's original request:
{state['question']}

══════ VALIDATION TARGETS ══════
EXPECTED NUM_DAYS: {num_days} — count the days in the draft and flag if different.
EXPECTED TRAVEL STYLE: {travel_style} — flag any recommendation that doesn't match.
══════════════════════════════════

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
    num_days = state['plan']['num_days']
    travel_style = state['plan'].get('travel_style', 'mid-range')

    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

══════ CRITICAL CONSTRAINTS ══════
NUMBER OF DAYS: {num_days} — write EXACTLY {num_days} day(s). Count them before finishing.
TRAVEL STYLE: {travel_style} — every recommendation must match this style.
══════════════════════════════════

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
