"""
Multi-Agent Travel Planner using LangGraph — Orchestrator Architecture
======================================================================
12 Agents with Central Coordinator (Hub-and-Spoke Pattern):
  Planner → Flight → Hotel → Visa → Weather → Activities → Places
  → Budget → ★ COORDINATOR ★ → Writer → Reviewer → Finalizer

Each research agent passes context to the next (inter-agent communication).
The Coordinator cross-validates ALL research before the Writer uses it.

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
    user_budget: Optional[float] = Field(None, description="Budget the user specified")
    estimated_total: Optional[float] = Field(None, description="Total estimated cost")
    is_over_budget: bool = Field(False)
    overage_amount: Optional[float] = Field(None)
    remaining_amount: Optional[float] = Field(None, description="Budget remaining after plan total")
    remaining_percentage: Optional[float] = Field(None, description="Remaining as % of total budget (e.g. 62 means 62% unused)")
    budget_underutilized: bool = Field(False, description="TRUE if luxury/mid-range traveler has >40% budget remaining — means plan picked too-cheap options")
    upgrade_suggestions: List[str] = Field(default_factory=list, description="Specific upgrades to better utilize the budget")
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
    wrongly_marked_domestic: bool = Field(False, description="TRUE if plan says 'no visa needed' but traveler is international (e.g. Saudi going to Egypt)")
    details: List[str] = Field(default_factory=list)


class SafetyCheck(BaseModel):
    travel_warnings_mentioned: bool = Field(False)
    insurance_mentioned: bool = Field(False)
    emergency_contacts_included: bool = Field(False)
    details: List[str] = Field(default_factory=list)


class TransportRealismCheck(BaseModel):
    """Validates that transport prices are realistic and match real-world costs."""
    transport_price_realistic: bool = Field(True, description="Is the transport price within realistic range?")
    reported_price: Optional[float] = Field(None, description="Transport price shown in the plan")
    realistic_min: Optional[float] = Field(None, description="Minimum realistic price for this route")
    realistic_max: Optional[float] = Field(None, description="Maximum realistic price for this route")
    is_round_trip: bool = Field(True, description="Is the price for round trip (not one-way)?")
    price_matches_style: bool = Field(True, description="Does the chosen price match the travel style? Luxury should NOT pick cheapest")
    details: List[str] = Field(default_factory=list)


class HotelOrderingCheck(BaseModel):
    """Validates hotel presentation order and selection logic."""
    ordered_expensive_first: bool = Field(True, description="Are hotels ordered from most expensive to cheapest?")
    recommended_matches_style: bool = Field(True, description="Does the AI recommendation match travel style? Luxury = most expensive")
    recommended_hotel_name: str = Field("", description="Name of the recommended hotel")
    recommended_price_per_night: Optional[float] = Field(None, description="Price per night of recommended hotel")
    most_expensive_available: Optional[float] = Field(None, description="Most expensive hotel option available")
    budget_allows_upgrade: bool = Field(False, description="Could the budget afford a more expensive hotel?")
    details: List[str] = Field(default_factory=list)


class PriceSelectionCheck(BaseModel):
    """Validates that price selections match travel style — not always cheapest."""
    flight_picked_cheapest_unnecessarily: bool = Field(False, description="Did it pick cheapest flight when budget allows better?")
    hotel_picked_cheapest_unnecessarily: bool = Field(False, description="Did it pick cheapest hotel when budget allows better?")
    budget_remaining_after_upgrades: Optional[float] = Field(None, description="Would budget still have room if we picked better options?")
    style_appropriate_selections: bool = Field(True, description="Do all selections match the travel style?")
    details: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    budget_check: BudgetCheck = Field(default_factory=BudgetCheck)
    logistics_check: LogisticsCheck = Field(default_factory=LogisticsCheck)
    visa_check: VisaCheck = Field(default_factory=VisaCheck)
    safety_check: SafetyCheck = Field(default_factory=SafetyCheck)
    transport_realism: TransportRealismCheck = Field(default_factory=TransportRealismCheck)
    hotel_ordering: HotelOrderingCheck = Field(default_factory=HotelOrderingCheck)
    price_selection: PriceSelectionCheck = Field(default_factory=PriceSelectionCheck)
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
    coordinator_brief: Optional[str]          # ← NEW: unified brief from Coordinator
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

═══════════════════════════════════════════════════
PRICE REALISM — EXTREMELY CRITICAL:
═══════════════════════════════════════════════════
- ALWAYS quote ROUND TRIP (return) prices, not one-way. State clearly "رايح وجاي" / "round trip".
- Domestic flights within Egypt cost AT LEAST 2,000-4,000 EGP round trip economy (2024-2026 prices).
  Examples: Cairo→Hurghada round trip = 3,000-8,000 EGP, Cairo→Sharm = 3,000-7,000 EGP,
  Cairo→Luxor = 2,500-6,000 EGP, Cairo→Aswan = 3,000-7,000 EGP.
- A flight ticket is NEVER 100-500 EGP. That is taxi fare, not a flight. If your search returns
  a price under 1,000 EGP for a domestic flight, the data is WRONG — search again or use known ranges.
- International flights from Egypt start at 8,000+ EGP minimum (nearby countries) and 15,000+ EGP for long-haul.
- Private car/transfer Cairo→Hurghada (5-6 hours) costs 2,000-4,000 EGP one way.
- Bus Cairo→Hurghada costs 300-600 EGP one way.
- ALWAYS search Tavily to verify prices. If Tavily returns no clear price, use the realistic ranges above.
- NEVER fabricate a suspiciously low price. When in doubt, quote the HIGHER end of the realistic range.

═══════════════════════════════════════════════════
PRICE SELECTION BY TRAVEL STYLE — DON'T ALWAYS PICK CHEAPEST:
═══════════════════════════════════════════════════
- LUXURY travelers: Recommend the BEST/MOST COMFORTABLE option, NOT the cheapest.
  Business class if available. Direct flights over cheaper connecting flights.
  If budget allows a 12,000 EGP business class and there's a 5,000 EGP economy, recommend BUSINESS.
  The traveler wants comfort and premium experience, not savings.
- MID-RANGE travelers: Recommend the MIDDLE option — good comfort at reasonable price.
  Economy class on good airlines, not the absolute cheapest budget carrier.
- BUDGET travelers: Recommend the cheapest practical option.
- NEVER default to "السعر الأدنى" (cheapest price) for luxury or mid-range travelers.
- The AI recommendation should match the travel style. A luxury traveler with 100K budget
  should NOT be told "أرخص رحلة هي 3,000 جنيه" — they should be told "أفضل رحلة بيزنس 12,000 جنيه".

OUTPUT RULES — VERY IMPORTANT:
- ONLY include transport/flight information: airlines, routes, prices, duration, tips.
- DO NOT calculate total trip budget or budget breakdown — that is the Budget agent's job.
- DO NOT include hotel costs, activity costs, or food costs.
- ONLY write about how to GET THERE and GET BACK.
- After searching, formulate bullet-point notes with transport options, prices, and duration.
- State whether each price is ONE-WAY or ROUND TRIP. Prefer quoting ROUND TRIP.
- ALWAYS present 3 transport options ordered from MOST EXPENSIVE to CHEAPEST.
- Your AI RECOMMENDATION must match the travel style:
  LUXURY → recommend Option 1 (most expensive/comfortable)
  MID-RANGE → recommend Option 2 (middle)
  BUDGET → recommend Option 3 (cheapest)"""

HOTEL_SYSTEM = """You are the Hotel agent. You have access to a web search tool called Tavily.

CRITICAL RULES:
1. LOCATION: Search ONLY in the EXACT destination. Never show a hotel from a different city.

2. TRAVEL STYLE DETERMINES HOTEL SELECTION:
   - LUXURY: Focus on 5-star, premium resorts. Your TOP recommendation MUST be the most luxurious option.
     Do NOT recommend "best value" — the traveler wants premium experience, not savings.
     If budget allows a 5-star all-inclusive at 10,000/night and there's a 4-star at 4,000/night,
     recommend the 5-star as #1 choice.
   - MID-RANGE: Balance quality and price. Recommend good 4-star hotels.
   - BUDGET: Focus on affordable, clean, well-rated budget hotels.

3. HOTEL TYPE INTELLIGENCE (meal plans):
   - ALL-INCLUSIVE: covers all meals + drinks + most activities. Outside spending = minimal.
   - FULL BOARD: covers breakfast + lunch + dinner. Outside spending = activities + transport only.
   - HALF BOARD: covers breakfast + dinner. Lunch outside needed.
   - BED & BREAKFAST: covers breakfast only. Lunch + dinner outside needed.

4. REALISTIC PRICING FOR SHORT TRIPS:
   - For 1-2 day trips: use ACTUAL hotel prices per night, not percentage-based allocation.
     A realistic luxury hotel in Egypt costs 3,000-15,000 EGP/night, NOT 45,000 EGP/night.
   - For 3+ day trips: max per night = total budget × hotel_percentage / number of nights
     Hotel percentages: All-Inclusive 75%, Full Board 65%, Half Board 55%, B&B 50%
   - NEVER recommend a per-night price that exceeds what real hotels actually charge.
   - Search Tavily to verify ACTUAL prices, don't just calculate from percentages.

5. Always provide EXACTLY 3 options ORDERED FROM MOST EXPENSIVE TO CHEAPEST:
   - LUXURY style: Option 1 = most luxurious & expensive, Option 2 = premium, Option 3 = upscale
     ALL must be luxury-tier. The FIRST option is the TOP recommendation.
   - MID-RANGE style: Option 1 = premium, Option 2 = mid-range, Option 3 = best value
   - BUDGET style: Option 1 = mid-range, Option 2 = budget-mid, Option 3 = cheapest good
   - ALL three must stay within what the budget allows for hotels.
   - ORDERING IS CRITICAL: Option 1 is ALWAYS the most expensive. NEVER put the cheapest first.

For each hotel:
  * Exact name (verified in destination via Tavily search)
  * Star rating
  * Price per night in preferred currency (VERIFIED real price, not calculated)
  * What is INCLUDED (all-inclusive / half-board / room only / breakfast)
  * Location within destination
  * What makes it special (beach access, pool, spa, etc.)
  * Rating if available
  * Best for (families, couples, etc.)

6. RECOMMENDATION LOGIC — MAXIMIZE QUALITY WITHIN BUDGET:
   - LUXURY travelers: recommend the MOST EXPENSIVE hotel the budget allows.
     If budget is 100,000 EGP for 2 nights and the most luxurious hotel costs 20,000/night (40,000 total),
     recommend THAT hotel — NOT a cheaper 15,000/night hotel. The traveler WANTS premium.
     Reason: "best luxury experience". NEVER "best value" for luxury travelers.
   - MID-RANGE travelers: recommend best quality/price balance.
   - BUDGET travelers: recommend best value for money.
   - NEVER recommend a cheaper hotel to a luxury traveler "because it balances price and service".
   - ALWAYS pick the most expensive option that fits within the total budget (not just hotel allocation).

- Use Tavily to search and verify real hotels and their ACTUAL prices in the destination.
- Show budget calculation before the 3 options.
- IMPORTANT: The hotel you recommend here will appear in ALL sections of the plan.
  Make sure your recommendation matches what a traveler of this style actually wants."""

VISA_SYSTEM = """You are the Visa agent. You have access to a web search tool called Tavily.

═══════════════════════════════════════════════════
STEP 1 — DETERMINE IF THIS IS DOMESTIC OR INTERNATIONAL TRAVEL:
═══════════════════════════════════════════════════
You will receive NATIONALITY and DESTINATION COUNTRY explicitly. Compare them:

DOMESTIC (same country) — NO visa needed:
  - Egyptian (مصري) traveling to anywhere in Egypt (شرم الشيخ، الغردقة، أسوان، الأقصر) → domestic
  - Saudi (سعودي) traveling to anywhere in Saudi Arabia (جدة، الرياض، أبها) → domestic
  - American traveling to anywhere in USA → domestic

INTERNATIONAL (different country) — visa research REQUIRED:
  - Saudi (سعودي) traveling to Egypt (شرم الشيخ، الغردقة، القاهرة) → INTERNATIONAL, visa needed!
  - Egyptian (مصري) traveling to Saudi Arabia → INTERNATIONAL, visa needed!
  - Egyptian traveling to Turkey, UAE, Europe → INTERNATIONAL, visa needed!

CRITICAL RULE: "سعودي" means Saudi Arabia. "شرم الشيخ" is in EGYPT. Saudi ≠ Egypt → VISA REQUIRED.
DO NOT confuse the nationality with the destination country. They are SEPARATE fields.
The fact that both are Arab countries does NOT mean they are the same country.

If DOMESTIC travel:
  Respond with exactly: "لا حاجة لتأشيرة — المسافر داخل بلده." and STOP.

═══════════════════════════════════════════════════
STEP 2 — IF INTERNATIONAL, RESEARCH VISA REQUIREMENTS:
═══════════════════════════════════════════════════
- Search specifically for visa requirements for THAT nationality passport traveling to THAT destination.
- Never give generic visa info — always specify the passport nationality.
- Cover: visa type, required documents, fees, processing time, e-visa options.
- Mention passport validity requirements (usually 6 months beyond travel dates).
- Include visa-on-arrival options if available for that nationality.
- Warn about common rejection reasons specific to that nationality.
- Use Tavily for current, accurate info.

COMMON EXAMPLES (for reference):
- Saudi → Egypt: Visa on arrival available, ~25 USD, or e-visa.
- Egyptian → Saudi: Visa required (Umrah, work, visit visa types).
- Egyptian → Turkey: E-visa available.
- Egyptian → UAE: Visa required.

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

CRITICAL — USE EXACT NUM_DAYS:
- The number of days is provided in the query. Use EXACTLY that number.
- If the user says 1 day, calculate budget for 1 day ONLY.
- NEVER calculate for more or fewer days than specified.

═══════════════════════════════════════════════════
SHORT TRIP INTELLIGENCE (1-3 days) — VERY IMPORTANT:
═══════════════════════════════════════════════════
For short trips (1-3 days), percentage-based allocation produces UNREALISTIC numbers.
Example: 60,000 EGP budget × 75% = 45,000 for hotel. But NO hotel costs 45,000/night!

For 1-3 day trips, use REAL PRICES instead of percentages:
1. Search Tavily for ACTUAL hotel prices per night at the destination.
2. Search for ACTUAL flight/transport costs.
3. Search for ACTUAL activity costs.
4. Add them up. The total should be WELL UNDER the budget for short luxury trips.
5. Report the ACTUAL remaining budget — do NOT force the total to equal the budget.

Example for 1-day luxury trip to Sahl Hasheesh, budget 60,000 EGP:
- Hotel (1 night, 5-star all-inclusive): ~8,000-15,000 EGP (real price)
- Flights (round trip): ~4,000-5,000 EGP
- Activities: ~3,000-5,000 EGP
- Transport (airport transfer): ~800-1,500 EGP
- Misc: ~2,000 EGP
- TOTAL: ~18,000-28,000 EGP → Remaining: 32,000-42,000 EGP
- This is CORRECT — a 1-day trip simply doesn't cost 60,000 EGP.

═══════════════════════════════════════════════════
LONGER TRIPS (4+ days) — use percentage allocation:
═══════════════════════════════════════════════════
PRIORITY ORDER (MUST follow this order):
  1st priority: HOTEL (الفندق أولاً — gets the biggest share)
  2nd priority: TRANSPORT (النقل ثانياً — flights/transfers are expensive)
  3rd priority: ACTIVITIES (الأنشطة ثالثاً — what's left after hotel + transport)
  4th: Miscellaneous

SMART ALLOCATION based on hotel meal plan:
- ALL-INCLUSIVE: Hotel 65% | Transport 15% | Activities 15% | Misc 5%
- FULL BOARD: Hotel 55% | Transport 20% | Activities 15% | Food 5% | Misc 5%
- HALF BOARD: Hotel 50% | Transport 20% | Activities 15% | Food 10% | Misc 5%
- BED & BREAKFAST: Hotel 45% | Transport 20% | Activities 15% | Food 15% | Misc 5%

BUDGET COMPARISON — MATH MUST BE CORRECT:
- Add up ALL category costs to get TOTAL.
- Remaining = User's budget - TOTAL.
- If remaining > 0: say "الميزانية المتبقية: X" (remaining budget: X).
- If remaining = 0: say "تم استخدام الميزانية بالكامل".
- If remaining < 0: CRITICAL ERROR — reduce costs to stay under budget.
- NEVER say "تم استهلاك الميزانية بالكامل" if the total is LESS than the budget.

═══════════════════════════════════════════════════
MAXIMIZE QUALITY WITHIN BUDGET — DON'T ALWAYS PICK CHEAPEST:
═══════════════════════════════════════════════════
- LUXURY travelers: Use the MOST EXPENSIVE hotel and transport options that fit in the budget.
  If hotel agent offers 3 hotels (20,000/night, 15,000/night, 10,000/night) and budget allows 20,000,
  USE the 20,000 option. NEVER pick 10,000 "to save money" — the luxury traveler WANTS the best.
  Same for flights: if business class fits in budget, USE business class.
- MID-RANGE travelers: Use the MIDDLE option from each agent's recommendations.
- BUDGET travelers: Use the cheapest options.
- NEVER default to cheapest across all styles. The budget allocation should MAXIMIZE quality
  within the total budget, not minimize spending.
- After calculating total, check the REMAINING BUDGET:
  remaining = user_budget - total
  remaining_percentage = (remaining / user_budget) × 100

  BUDGET UTILIZATION RULES:
  - LUXURY traveler with >40% remaining: THIS IS A PROBLEM. You're not using the budget well.
    The traveler gave 200,000 and you're only spending 76,500? That means you picked cheap options.
    → Recommend upgrading: better hotel suite, business class, premium activities, spa packages.
    → Calculate what the total WOULD BE with upgrades and show both options.
  - LUXURY traveler with 20-40% remaining: Acceptable but suggest optional upgrades.
  - LUXURY traveler with <20% remaining: Perfect utilization.
  - MID-RANGE traveler with >50% remaining: Suggest upgrades to improve the experience.
  - BUDGET traveler: Any remaining amount is fine — they want to save.

  IMPORTANT: "remaining budget" is sometimes GOOD (short trips, budget travelers) but for LUXURY
  travelers with large budgets, huge remaining amounts mean the plan is UNDERPERFORMING.

CROSS-REFERENCE WITH OTHER AGENTS:
- You will receive hotel_notes and flight_notes from other agents.
- Use the ACTUAL hotel and flight prices from those notes in your budget.
- The hotel price in your budget MUST match the hotel recommended by the Hotel agent.
- The transport cost MUST match the flight/transport prices from the Flight agent.
- DO NOT invent different prices — use the researched prices.
- Use the AI-RECOMMENDED option from each agent (which should match the travel style).

═══════════════════════════════════════════════════
NUMBER OF TRAVELERS — MULTIPLY EVERYTHING:
═══════════════════════════════════════════════════
- You will receive NUM_TRAVELERS in the query. This is the number of people traveling.
- Transport costs from the Flight agent are PER PERSON. You MUST multiply by NUM_TRAVELERS.
  Example: Flight agent says "13,500 EGP per person round trip", NUM_TRAVELERS = 4
  → Transport budget = 13,500 × 4 = 54,000 EGP (NOT 13,500!)
- Hotel costs are USUALLY per room, NOT per person. Do NOT multiply hotel by travelers
  unless the hotel price is explicitly "per person" (e.g. all-inclusive per person).
- Activity costs: multiply by NUM_TRAVELERS if priced per person.
- Food costs: multiply by NUM_TRAVELERS.
- CRITICAL: The transport total in your budget MUST equal per_person_cost × NUM_TRAVELERS.
  If your budget shows a different transport number than the main transport table,
  the plan will be REJECTED. These numbers MUST match exactly.

RULES:
- Total must NEVER exceed the budget.
- Never add food budget if hotel is all-inclusive or full board.
- Show final breakdown table clearly.
- Use preferred currency for ALL prices.
- Use Tavily for current prices to verify.
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
- Emergency contacts: nearest hospital, police number
  IMPORTANT: Do NOT include embassy info if traveler is traveling WITHIN their own country.
  An Egyptian traveling inside Egypt does NOT need the Egyptian embassy!
  Only include embassy info for INTERNATIONAL travel.

FOR HOTELS — present the 3 hotels ONCE ONLY in a comparison table, ORDERED FROM MOST EXPENSIVE TO CHEAPEST:
| الفندق | النجوم | السعر/ليلة | المميزات | التقييم |
|--------|--------|------------|----------|---------|
| فندق 1 (الأغلى) | ⭐⭐⭐⭐⭐ | X ج.م | كذا | 9.2 |
| فندق 2 | ⭐⭐⭐⭐⭐ | X ج.م | كذا | 8.8 |
| فندق 3 (الأرخص) | ⭐⭐⭐⭐ | X ج.م | كذا | 8.5 |
CRITICAL: Option 1 is ALWAYS the most expensive. The table goes from highest price to lowest.
Then write: "🏆 توصية الـ AI: [اسم الفندق الأغلى] — لأن [السبب]"
For LUXURY travelers, the AI recommendation MUST be the MOST EXPENSIVE hotel (Option 1).
IMPORTANT: Do NOT repeat or mention hotels anywhere else in the plan.

FOR FLIGHTS/TRANSPORT — present ALL options in a table ONCE ONLY (NO links):
| وسيلة النقل | المدة | التكلفة/فرد | الإجمالي (عدد المسافرين) | الملاحظات |
|------------|-------|------------|------------------------|-----------|
| الخيار 1 (الأفضل) | Xhr | X ج.م | X × عدد = Y ج.م | رحلة رايح وجاي، بيزنس |
| الخيار 2 | Xhr | X ج.م | X × عدد = Y ج.م | رايح وجاي، اقتصادي |
| الخيار 3 (الأرخص) | Xhr | X ج.م | X × عدد = Y ج.م | رايح وجاي |
CRITICAL RULES:
- ALWAYS show price PER PERSON and TOTAL for all travelers.
- If 3 travelers and flight costs 8,000/person, write "8,000 ج.م" under التكلفة/فرد and "24,000 ج.م" under الإجمالي.
- ALWAYS specify if the price is round trip (رايح وجاي) or one-way.
- Show ALL 3 transport options, ordered from most expensive/comfortable to cheapest.
- The AI recommendation line below the table should match the travel style.
- Write: "🏆 توصية الـ AI: [الخيار] — لأن [السبب]"
IMPORTANT: Do NOT repeat transport options anywhere else.

═══════════════════════════════════════════════════
BUDGET SECTION — PLACEMENT RULES:
═══════════════════════════════════════════════════
- Put the FULL budget breakdown at the END of the plan under "💰 الميزانية" or "💰 Budget".
- DO NOT put budget totals or breakdowns inside the transport/flights section.
- The transport section should ONLY show transport option prices, NOT total trip budget.
- Budget section should show: transport cost + hotel cost + activities cost + food cost = TOTAL
  and compare to user's budget.

═══════════════════════════════════════════════════
BUDGET MATH — MUST BE CORRECT:
═══════════════════════════════════════════════════
- Add up ALL costs EXACTLY. Use the ACTUAL numbers from the budget research notes.
- Remaining budget = User's total budget - Your calculated total.
- If remaining > 0: write "الميزانية المتبقية: [remaining amount]". NEVER say "تم استهلاك الميزانية".
- If remaining = 0 exactly: write "تم استخدام الميزانية بالكامل".
- If total > budget: ERROR — reduce costs.
- For SHORT TRIPS (1-3 days): the total will likely be MUCH LESS than the budget. This is NORMAL.
  A 1-day trip with 60,000 budget may only cost 20,000-30,000. Report the real remaining amount.

═══════════════════════════════════════════════════
TRANSPORT IN BUDGET — MULTIPLY BY TRAVELERS:
═══════════════════════════════════════════════════
- Transport prices in the transport table are PER PERSON.
- In the budget section, the transport cost MUST be the TOTAL for ALL travelers.
- If transport = 13,500/person and there are 4 travelers → budget transport = 54,000 (NOT 13,500).
- The budget transport number MUST MATCH the "الإجمالي" column in the transport table.
- If these numbers don't match, the plan is WRONG. Fix it before writing.

═══════════════════════════════════════════════════
CONSISTENCY — CRITICAL:
═══════════════════════════════════════════════════
- The hotel in your budget table MUST be the SAME hotel shown in the hotels section.
- The transport cost in budget MUST match the TOTAL transport (per_person × num_travelers) from the transport table.
- Do NOT show one hotel in the hotels table and a different hotel in the budget notes.
- The AI recommendation hotel = the hotel used for budget calculation = the hotel in the itinerary.

DO NOT include a "نظرة عامة على الرحلة" (trip overview) section. This information is already
displayed in the app header (destination, dates, travelers, budget). Starting the plan with
an overview repeats what the user already sees. Start directly with the hotels/transport tables.

Use ALL research notes: flights/transport, hotels, visa, weather, activities, places, budget.
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

## Visa validation — CRITICAL FOR INTERNATIONAL TRAVEL
- If traveler nationality is DIFFERENT from destination country, visa info MUST be present.
  Example: Saudi (سعودي) going to Egypt (شرم الشيخ) → visa info REQUIRED. If missing: -20 points.
- "لا حاجة لتأشيرة — المسافر داخل بلده" is ONLY correct if nationality matches destination country.
  Saudi → Egypt is NOT domestic! Egyptian → Saudi is NOT domestic!
- Complete visa info? Passport validity? Common pitfalls?
- If plan says "no visa needed" for an international traveler: CRITICAL ERROR, -25 points.

## Weather validation
- Activities appropriate for weather? Packing list matches weather?

## Safety validation
- Travel warnings? Insurance? Emergency contacts?

## Hotel consistency validation
- The recommended hotel MUST be the same in the hotels section and the budget section.
- If different hotels are mentioned in different sections, flag: -15 points.
- For luxury travelers, the recommended hotel should be the MOST luxurious, not "best value".
- ALL 3 hotel options MUST be shown in the hotels table. If only 1 hotel is shown: -10 points.

## Transport display validation
- Transport prices MUST show both per-person AND total for all travelers.
- If 3 travelers and only total is shown without per-person breakdown: -5 points.
- All transport prices must specify if round trip (رايح وجاي) or one-way: -5 if missing.

## Budget math validation
- Add up the budget categories. Total must equal what's reported.
- Remaining budget = user budget - total. If remaining > 0 but plan says "fully consumed", flag: -15 points.
- For short trips (1-3 days), if hotel cost seems unrealistically high (e.g., 45,000/night), flag: -15 points.
- Transport cost in budget must match actual flight prices from research.
- Budget priority must be: Hotel first (biggest share) → Transport second → Activities third.

## Transport × Travelers consistency — CRITICAL:
- Transport table shows per-person cost and total (per_person × travelers).
- Budget section transport MUST equal that total, NOT the per-person amount.
- If transport table says "13,500/person × 4 = 54,000" but budget says "15,250 for transport": CRITICAL ERROR, -25 points.
- These numbers MUST match. If they don't, add to fix_instructions.
- Also check: activities and food in budget should be multiplied by num_travelers if priced per-person.

## Budget utilization validation — CRITICAL FOR LUXURY/MID-RANGE:
- Calculate: remaining_percentage = (remaining / user_budget) × 100
- Fill budget_check.remaining_amount and budget_check.remaining_percentage.
- LUXURY traveler with remaining_percentage > 40%: Set budget_check.budget_underutilized = TRUE.
  This is a CRITICAL issue: -20 points. The plan is picking cheap options for a luxury traveler.
  Example: 200,000 budget, 76,500 total, 123,500 remaining (62%) → UNACCEPTABLE for luxury.
  Add to fix_instructions: "Budget underutilized (X% remaining). Upgrade hotel to most expensive option,
  use business class flights, add premium activities. Target: use at least 60% of budget for luxury."
  Add to budget_check.upgrade_suggestions: specific upgrades (better hotel, business class, etc.)
- MID-RANGE traveler with remaining_percentage > 50%: budget_underutilized = TRUE, -10 points.
- BUDGET traveler: budget_underutilized = FALSE regardless of remaining. They WANT to save.
- Short trips (1-3 days) with BUDGET style: large remaining is expected and acceptable.

## Transport price realism validation
- Domestic flights in Egypt: minimum 2,000-4,000 EGP round trip. NEVER under 1,000 EGP.
- If flight price is under 1,000 EGP (e.g., 185 EGP), flag as UNREALISTIC: -20 points.
- Private car transfers 5-6 hours: 2,000-4,000 EGP one way. Under 500 EGP is UNREALISTIC.
- All transport prices should be ROUND TRIP unless clearly stated as one-way.

## Hotel ordering validation
- Hotels MUST be ordered from most expensive to cheapest (Option 1 = most expensive).
- If cheapest hotel is listed first, flag: -10 points.
- For LUXURY travelers: the recommended hotel MUST be the most expensive option. Flag if not: -15 points.

## Domestic travel validation
- If traveler is traveling within their own country, there should be NO embassy info. Flag if present: -10 points.

## STRUCTURED OUTPUT — YOU MUST FILL THESE FIELDS:

### transport_realism (TransportRealismCheck):
- transport_price_realistic: Is the transport price within the realistic range? (min 2,000 EGP domestic flight)
- reported_price: The actual transport price shown in the plan.
- realistic_min: Minimum realistic price for this route (e.g., 3,000 for Cairo→Hurghada).
- realistic_max: Maximum realistic price for this route (e.g., 8,000 for Cairo→Hurghada).
- is_round_trip: Is the quoted price for round trip? Must be true.
- price_matches_style: Does the chosen price match the travel style?
  LUXURY should NOT pick the cheapest flight. MID-RANGE should pick middle. BUDGET picks cheapest.
- details: List any issues found.

### hotel_ordering (HotelOrderingCheck):
- ordered_expensive_first: Are the 3 hotels ordered from MOST expensive to CHEAPEST? Option 1 must be most expensive.
- recommended_matches_style: For LUXURY → recommended = most expensive. For BUDGET → recommended = cheapest.
- recommended_hotel_name: Name of the AI-recommended hotel.
- recommended_price_per_night: Its price per night.
- most_expensive_available: Price of the most expensive hotel option listed.
- budget_allows_upgrade: Could the budget afford a more expensive hotel than what was recommended?
- details: List any issues found.

### price_selection (PriceSelectionCheck):
- flight_picked_cheapest_unnecessarily: TRUE if travel style is luxury/mid-range but cheapest flight was recommended.
- hotel_picked_cheapest_unnecessarily: TRUE if travel style is luxury/mid-range but cheapest hotel was recommended.
- budget_remaining_after_upgrades: If we picked better options matching the style, would budget still work?
- style_appropriate_selections: Do ALL selections (hotel, flight, activities) match the travel style?
- details: List any issues found.

## Scoring: Start at 100, deduct:
- Day count mismatch: -25 | Travel style mismatch: -15
- Budget exceeded: -20 | Missing visa: -15 | Unrealistic schedule: -10
- No transit info: -10 | No airport transfer: -5 | No safety info: -5
- Weather mismatch: -5 | Each hallucination risk: -5
- Budget in wrong section: -10 | Hotel inconsistency: -15
- Wrong remaining budget: -15 | Embassy for domestic travel: -10
- Unrealistic hotel price: -15 | Unrealistic transport price: -20
- Hotels ordered wrong (cheapest first): -10
- Luxury traveler but cheapest hotel recommended: -15
- Budget priorities wrong (activities before transport): -10
- Cheapest flight picked for luxury traveler: -15
- Price selections don't match travel style: -10
- Budget underutilized (luxury >40% remaining): -20
- Budget underutilized (mid-range >50% remaining): -10
- Only 1 hotel shown instead of 3: -10
- Transport missing per-person price: -5

CRITICAL: If transport_realism.transport_price_realistic is False → score MUST be < 80.
CRITICAL: If hotel_ordering.recommended_matches_style is False for luxury → score MUST be < 80.
CRITICAL: If price_selection.flight_picked_cheapest_unnecessarily is True for luxury → score MUST be < 80.
CRITICAL: If budget_check.budget_underutilized is True for luxury → score MUST be < 80.
These ensure the plan gets sent BACK to the Writer for fixes.

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

DO NOT include a "نظرة عامة على الرحلة" (trip overview) section. The app header already
shows destination, dates, number of travelers, and budget. Do NOT repeat this info.
Start directly with the hotel and transport comparison tables.

Your output must include:
1. HOTELS — ALWAYS show ALL 3 hotel options in a comparison table:
| الفندق | النجوم | السعر/ليلة | المميزات | التقييم |
|--------|--------|------------|----------|---------|
| فندق 1 (الأغلى) | ⭐⭐⭐⭐⭐ | X ج.م | كذا | 9.2 |
| فندق 2 | ⭐⭐⭐⭐⭐ | X ج.م | كذا | 8.8 |
| فندق 3 (الأرخص) | ⭐⭐⭐⭐ | X ج.م | كذا | 8.5 |
Then: "🏆 توصية الـ AI: [اسم الفندق] — لأن [السبب]"
CRITICAL: Show ALL 3 hotels. NEVER show just 1 hotel. Ordered from most expensive to cheapest.

3. TRANSPORT — ALWAYS show ALL options with per-person AND total price:
| وسيلة النقل | المدة | التكلفة/فرد | الإجمالي (عدد المسافرين) | الملاحظات |
|------------|-------|------------|------------------------|-----------|
| الخيار 1 | Xhr | X ج.م/فرد | X × عدد = Y ج.م | رايح وجاي |
CRITICAL: Always show price PER PERSON + TOTAL. If 3 travelers × 8,000 = write "24,000 ج.م إجمالي".
Then: "🏆 توصية الـ AI: [الخيار] — لأن [السبب]"

4. Day-by-day itinerary with times, places, costs (EXACTLY num_days days)
5. Practical info (transport, money, language, safety)
6. Packing checklist (based on weather)
7. Emergency info (hospital, police numbers)
   IMPORTANT: Do NOT include embassy info for DOMESTIC travel.
   If the traveler is traveling within their own country (e.g., Egyptian in Egypt), NO embassy needed.
   Only include embassy for international travel.
8. Budget breakdown at the END (total by category vs user budget)

If a review exists, incorporate ALL fixes.
Make it ready to print and follow.

BUDGET PLACEMENT:
- The complete budget summary goes at the END under "💰 الميزانية" or "💰 Budget".
- Transport section shows only transport options and prices.
- NEVER put the full budget breakdown inside the transport/flights section.

BUDGET MATH — MUST BE CORRECT:
- Add costs exactly. Remaining = budget - total.
- If remaining > 0: write "الميزانية المتبقية: [amount]". NEVER say "تم استهلاك الميزانية".
- For short trips (1-3 days), total will be MUCH less than budget. This is normal and correct.
- The hotel in budget MUST match the recommended hotel in the hotels section.
- Transport cost in budget MUST be the TOTAL for ALL travelers (per_person × num_travelers).
  If transport table shows 13,500/person × 4 travelers = 54,000, budget transport = 54,000.
  NOT 13,500. The budget transport MUST match the "الإجمالي" column in the transport table.

CONSISTENCY CHECK BEFORE FINALIZING:
- Hotel name in hotels table = hotel in budget = hotel in itinerary (all same)
- Transport cost in budget = TOTAL transport from transport table (per_person × travelers)
- Total cost in budget < user's budget (with remaining amount shown)
- No embassy info for domestic travel
- No "نظرة عامة" section (info is in app header)

IMPORTANT: Do NOT include any confidence score, quality score, rating score,
or any numerical evaluation in your output. Never write phrases like
"درجة الثقة", "Confidence Score", "Quality Score", or any score number."""


COORDINATOR_SYSTEM = """You are the Orchestrator/Coordinator agent — the CENTRAL HUB of the travel planning system.
You receive ALL research from every agent and your job is to:
1. CROSS-VALIDATE all data for consistency
2. RESOLVE conflicts between agents
3. Produce a single UNIFIED BRIEF that the Writer will use

═══════════════════════════════════════════════════
STEP 1 — HOTEL VALIDATION (ORDER + MAXIMIZE QUALITY)
═══════════════════════════════════════════════════
- Extract ALL hotel options from hotel_notes.
- ORDERING: Hotels MUST be listed from MOST EXPENSIVE to CHEAPEST. If they aren't, REORDER them.
- Verify it matches the travel style:
  • LUXURY → must be a 5-star / premium resort. If hotel agent recommended a budget option, FLAG IT.
  • MID-RANGE → should be 4-star. Flag if 2-star or 5-star luxury.
  • BUDGET → should be affordable. Flag if it's a luxury resort.
- MAXIMIZE QUALITY: For LUXURY travelers, the RECOMMENDED hotel must be the MOST EXPENSIVE
  option that the budget allows. If budget is 100,000 EGP for 2 nights and there's a 20,000/night
  hotel (40,000 total), recommend THAT one — NOT a cheaper 15,000/night hotel.
  Calculate: can the budget cover (most_expensive_hotel × nights) + transport + activities?
  If YES → recommend the most expensive hotel.
- Record: HOTEL_NAME, HOTEL_PRICE_PER_NIGHT, HOTEL_TOTAL (price × nights), HOTEL_MEAL_PLAN

═══════════════════════════════════════════════════
STEP 2 — TRANSPORT VALIDATION (PRICE REALISM CRITICAL)
═══════════════════════════════════════════════════
- Extract transport option and cost from flight_notes.
- Verify it matches travel style:
  • LUXURY → must be flights or private transfer, NEVER public bus.
  • BUDGET → cheapest is fine.
- PRICE REALITY CHECK:
  • Domestic flights in Egypt: minimum 2,000-4,000 EGP round trip. NEVER under 1,000 EGP.
  • If transport cost seems unrealistically low (e.g., 185 EGP for a flight), FLAG IT and
    replace with realistic price range (e.g., Cairo→Hurghada round trip = 4,000-8,000 EGP).
  • Private car transfers: 2,000-4,000 EGP for 5-6 hour routes.
  • Always use ROUND TRIP prices in the budget.
- Record: TRANSPORT_TYPE, TRANSPORT_COST (round trip)

═══════════════════════════════════════════════════
STEP 3 — BUDGET CROSS-CHECK (MOST CRITICAL)
═══════════════════════════════════════════════════
- Take HOTEL_TOTAL from Step 1 and TRANSPORT_COST from Step 2.
- CRITICAL: Transport costs are PER PERSON. The user's query contains "Number of travelers: X".
  Extract that number. Multiply transport per-person cost by NUM_TRAVELERS to get TOTAL TRANSPORT.
  Example: flight = 13,500/person, 4 travelers → TOTAL TRANSPORT = 54,000. NOT 13,500.
- Compare with what the Budget agent calculated.
- If Budget agent used DIFFERENT hotel price → OVERRIDE with the real hotel price.
- If Budget agent used DIFFERENT transport cost → OVERRIDE with the real transport cost.
- If Budget agent did NOT multiply transport by num_travelers → FIX IT.
- PRIORITY ORDER for budget allocation:
  1st: HOTEL (الفندق) — gets the biggest share, maximize quality
  2nd: TRANSPORT (النقل) — flights/transfers are non-negotiable costs
  3rd: ACTIVITIES (الأنشطة) — what remains after hotel + transport
- Recalculate: TOTAL = hotel + transport(×travelers) + activities(×travelers) + food(×travelers) + misc
- REMAINING = user_budget - TOTAL
- If REMAINING > 0: mark as "under budget" with remaining amount.
- If REMAINING < 0: FLAG as over-budget and suggest cuts (cut activities first, NOT hotel).
- NEVER say "budget fully consumed" if remaining > 0.
- For SHORT TRIPS (1-3 days): total will naturally be MUCH less than budget. This is CORRECT.

═══════════════════════════════════════════════════
STEP 4 — DOMESTIC TRAVEL CHECK
═══════════════════════════════════════════════════
- If the traveler's nationality matches the destination country → mark DOMESTIC_TRAVEL = true
- If DOMESTIC_TRAVEL: no visa needed, NO embassy info, no passport requirements.
- Flag any research that incorrectly includes embassy/visa for domestic travel.

═══════════════════════════════════════════════════
STEP 5 — ACTIVITIES & PLACES VALIDATION
═══════════════════════════════════════════════════
- Verify all activities and places are in the CORRECT destination (not a different city).
- Check activities match travel style (no street food tours for luxury travelers).
- Remove duplicates between activities_notes and places_notes.
- Verify activity costs are reasonable.

═══════════════════════════════════════════════════
STEP 6 — WEATHER-ACTIVITY ALIGNMENT
═══════════════════════════════════════════════════
- Check weather_notes against activities_notes.
- Flag outdoor activities during extreme weather.
- Suggest alternatives if weather is bad for planned activities.

═══════════════════════════════════════════════════
OUTPUT FORMAT — UNIFIED BRIEF
═══════════════════════════════════════════════════
Produce a structured brief with these exact sections:

## 🎯 VALIDATED DATA
- Destination: [name]
- Dates: [dates]
- Duration: [X] days
- Travel Style: [luxury/mid-range/budget]
- Domestic Travel: [yes/no]

## ✈️ CONFIRMED TRANSPORT
- Type: [flight/bus/private transfer]
- Cost: [amount in currency]
- Details: [route, duration]

## 🏨 CONFIRMED HOTEL
- Name: [exact hotel name]
- Stars: [rating]
- Price/Night: [amount]
- Total ([X] nights): [amount]
- Meal Plan: [all-inclusive/half-board/etc.]
- What's Included: [details]

## 🎯 CONFIRMED ACTIVITIES
[List validated activities with costs]

## 📍 CONFIRMED PLACES
[List validated places]

## 💰 VALIDATED BUDGET
| Category | Cost |
|----------|------|
| Transport | X |
| Hotel ([X] nights) | X |
| Activities | X |
| Food (if needed) | X |
| Miscellaneous | X |
| **TOTAL** | **X** |
| **User Budget** | **X** |
| **Remaining** | **X** |

## ⚠️ CONFLICTS RESOLVED
[List any conflicts found and how they were resolved]

## 🛂 VISA STATUS
[Visa info or "domestic travel — no visa needed"]

## 🌤️ WEATHER SUMMARY
[Key weather points and packing advice]

This brief is the SINGLE SOURCE OF TRUTH for the Writer and Finalizer."""


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
    num_days = state['plan']['num_days']

    # ← ORCHESTRATOR PATTERN: Hotel receives flight context for transport-aware recommendations
    flight_context = ""
    if state.get('flight_notes'):
        flight_summary = chr(10).join('- ' + n for n in state['flight_notes'][:5])
        flight_context = f"\n\n══ CONTEXT FROM TRANSPORT AGENT ══\nThe traveler will arrive via:\n{flight_summary}\nConsider hotel proximity to arrival point (airport/bus station).\n══════════════════════════════════\n"

    output = _invoke_agent(agent,
        f"Find hotels for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}\nNumber of nights: {num_days}\nTravel style: {travel_style}\n{flight_context}\nIMPORTANT: This is a {num_days}-day trip. Search for REAL hotel prices per night. For luxury style, recommend the MOST LUXURIOUS option as your top pick, not 'best value'."
    )
    state["hotel_notes"] = _parse_notes(output)
    return state


def visa_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(VISA_SYSTEM)

    # ── Extract nationality explicitly from the question ──
    nationality = "unknown"
    for line in state['question'].split('\n'):
        line_stripped = line.strip()
        if 'nationality' in line_stripped.lower() or 'جنسي' in line_stripped:
            if ':' in line_stripped:
                nationality = line_stripped.split(':', 1)[1].strip()
            elif '：' in line_stripped:
                nationality = line_stripped.split('：', 1)[1].strip()
            break

    destination = state['plan']['destination']

    # ── Map nationality to country ──
    nationality_country_map = {
        'مصري': 'egypt', 'egyptian': 'egypt', 'مصرية': 'egypt',
        'سعودي': 'saudi arabia', 'saudi': 'saudi arabia', 'سعودية': 'saudi arabia',
        'إماراتي': 'uae', 'emirati': 'uae', 'إماراتية': 'uae',
        'أردني': 'jordan', 'jordanian': 'jordan',
        'لبناني': 'lebanon', 'lebanese': 'lebanon',
        'عراقي': 'iraq', 'iraqi': 'iraq',
        'كويتي': 'kuwait', 'kuwaiti': 'kuwait', 'كويتية': 'kuwait',
        'بحريني': 'bahrain', 'bahraini': 'bahrain',
        'عماني': 'oman', 'omani': 'oman',
        'قطري': 'qatar', 'qatari': 'qatar',
        'تونسي': 'tunisia', 'tunisian': 'tunisia',
        'مغربي': 'morocco', 'moroccan': 'morocco',
        'جزائري': 'algeria', 'algerian': 'algeria',
        'سوري': 'syria', 'syrian': 'syria',
        'أمريكي': 'usa', 'american': 'usa',
        'بريطاني': 'uk', 'british': 'uk',
        'فرنسي': 'france', 'french': 'france',
        'ألماني': 'germany', 'german': 'germany',
    }
    nat_lower = nationality.lower().strip()
    traveler_country = nationality_country_map.get(nat_lower, 'unknown')

    # ── Map destination (city/resort) to country — DETERMINISTIC ──
    # This is critical: the destination from the plan is a CITY name, not a country.
    # We must map it to the correct country to compare with traveler's country.
    destination_country_keywords = {
        'egypt': [
            'مصر', 'egypt', 'cairo', 'القاهرة', 'sharm', 'شرم', 'hurghada', 'الغردقة',
            'luxor', 'الأقصر', 'aswan', 'أسوان', 'dahab', 'دهب', 'marsa alam', 'مرسى علم',
            'el gouna', 'الجونة', 'ain sokhna', 'العين السخنة', 'سهل حشيش', 'sahl hasheesh',
            'alexandria', 'الإسكندرية', 'siwa', 'سيوة', 'nuweiba', 'نويبع', 'taba', 'طابا',
            'marsa matruh', 'مرسى مطروح', 'ras sudr', 'رأس سدر', 'port said', 'بورسعيد',
            'soma bay', 'سوما باي', 'makadi', 'مكادي', 'safaga', 'سفاجا', 'north coast', 'الساحل الشمالي',
        ],
        'saudi arabia': [
            'السعودية', 'saudi', 'riyadh', 'الرياض', 'jeddah', 'جدة', 'mecca', 'مكة',
            'medina', 'المدينة', 'abha', 'أبها', 'taif', 'الطائف', 'dammam', 'الدمام',
            'khobar', 'الخبر', 'neom', 'نيوم', 'al ula', 'العلا', 'yanbu', 'ينبع',
        ],
        'uae': [
            'الإمارات', 'uae', 'dubai', 'دبي', 'abu dhabi', 'أبو ظبي', 'sharjah', 'الشارقة',
            'ras al khaimah', 'رأس الخيمة', 'ajman', 'عجمان', 'fujairah', 'الفجيرة',
        ],
        'jordan': ['الأردن', 'jordan', 'amman', 'عمان', 'petra', 'البتراء', 'aqaba', 'العقبة', 'dead sea'],
        'lebanon': ['لبنان', 'lebanon', 'beirut', 'بيروت'],
        'turkey': ['تركيا', 'turkey', 'istanbul', 'إسطنبول', 'antalya', 'أنطاليا', 'bodrum', 'بودروم', 'trabzon', 'طرابزون'],
        'morocco': ['المغرب', 'morocco', 'marrakech', 'مراكش', 'casablanca', 'الدار البيضاء', 'fes', 'فاس', 'tangier', 'طنجة'],
        'tunisia': ['تونس', 'tunisia', 'tunis', 'sousse', 'سوسة', 'hammamet', 'الحمامات'],
        'qatar': ['قطر', 'qatar', 'doha', 'الدوحة'],
        'kuwait': ['الكويت', 'kuwait'],
        'bahrain': ['البحرين', 'bahrain', 'manama', 'المنامة'],
        'oman': ['عُمان', 'عمان', 'oman', 'muscat', 'مسقط', 'salalah', 'صلالة'],
        'iraq': ['العراق', 'iraq', 'baghdad', 'بغداد', 'erbil', 'أربيل'],
        'usa': ['أمريكا', 'usa', 'united states', 'new york', 'los angeles', 'miami', 'las vegas'],
        'uk': ['بريطانيا', 'uk', 'united kingdom', 'london', 'لندن', 'england'],
        'france': ['فرنسا', 'france', 'paris', 'باريس'],
        'germany': ['ألمانيا', 'germany', 'berlin', 'برلين', 'munich', 'ميونخ'],
        'italy': ['إيطاليا', 'italy', 'rome', 'روما', 'milan', 'ميلان'],
        'spain': ['إسبانيا', 'spain', 'barcelona', 'برشلونة', 'madrid', 'مدريد'],
        'greece': ['اليونان', 'greece', 'athens', 'أثينا', 'santorini', 'سانتوريني'],
        'malaysia': ['ماليزيا', 'malaysia', 'kuala lumpur', 'كوالالمبور'],
        'indonesia': ['إندونيسيا', 'indonesia', 'bali', 'بالي', 'jakarta', 'جاكرتا'],
        'thailand': ['تايلاند', 'thailand', 'bangkok', 'بانكوك', 'phuket', 'بوكيت'],
        'maldives': ['المالديف', 'maldives', 'malé', 'ماليه'],
    }

    dest_lower = destination.lower().strip()
    destination_country = 'unknown'
    for country, keywords in destination_country_keywords.items():
        for kw in keywords:
            if kw in dest_lower:
                destination_country = country
                break
        if destination_country != 'unknown':
            break

    # ── DETERMINISTIC domestic/international check ──
    # This decision is NOT left to the LLM — we compute it ourselves
    is_domestic = (traveler_country != 'unknown' and
                   destination_country != 'unknown' and
                   traveler_country == destination_country)

    if is_domestic:
        # Skip LLM entirely — we know this is domestic
        state["visa_notes"] = ["لا حاجة لتأشيرة — المسافر داخل بلده."]
        return state

    # ── INTERNATIONAL travel — force the LLM to research visa ──
    # Map traveler_country to Arabic name for display
    country_arabic = {
        'egypt': 'مصر', 'saudi arabia': 'السعودية', 'uae': 'الإمارات',
        'jordan': 'الأردن', 'lebanon': 'لبنان', 'iraq': 'العراق',
        'kuwait': 'الكويت', 'bahrain': 'البحرين', 'oman': 'عُمان',
        'qatar': 'قطر', 'tunisia': 'تونس', 'morocco': 'المغرب',
        'algeria': 'الجزائر', 'syria': 'سوريا', 'usa': 'أمريكا',
        'uk': 'بريطانيا', 'france': 'فرنسا', 'germany': 'ألمانيا',
    }
    traveler_country_ar = country_arabic.get(traveler_country, traveler_country)
    dest_country_ar = country_arabic.get(destination_country, destination_country)

    output = _invoke_agent(agent,
        f"""═══ VISA RESEARCH REQUEST — INTERNATIONAL TRAVEL ═══
⚠️ THIS IS CONFIRMED INTERNATIONAL TRAVEL. DO NOT say "no visa needed" or "domestic travel".

TRAVELER NATIONALITY: {nationality}
TRAVELER'S COUNTRY: {traveler_country_ar} ({traveler_country})
DESTINATION CITY: {destination}
DESTINATION COUNTRY: {dest_country_ar} ({destination_country})

{traveler_country_ar} ≠ {dest_country_ar} → THIS IS INTERNATIONAL TRAVEL.
The traveler NEEDS visa information. Research visa requirements NOW.

Search for: "{nationality} passport visa requirements for {destination_country}"

Original request: {state['question']}
═══════════════════════════════════════════════════"""
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

    # ← ORCHESTRATOR PATTERN: Activities receives hotel + weather context
    hotel_context = ""
    if state.get('hotel_notes'):
        hotel_summary = chr(10).join('- ' + n for n in state['hotel_notes'][:5])
        hotel_context = f"\n\n══ CONTEXT FROM HOTEL AGENT ══\nTraveler's hotel info:\n{hotel_summary}\nRecommend activities near the hotel area. If hotel is all-inclusive, focus on unique outside experiences.\n══════════════════════════════\n"

    weather_context = ""
    if state.get('weather_notes'):
        weather_summary = chr(10).join('- ' + n for n in state['weather_notes'][:5])
        weather_context = f"\n\n══ CONTEXT FROM WEATHER AGENT ══\nWeather conditions:\n{weather_summary}\nOnly recommend outdoor activities if weather permits. Suggest indoor alternatives for bad weather.\n══════════════════════════════════\n"

    output = _invoke_agent(agent,
        f"Find activities and experiences for: {state['question']}\nDestination: {state['plan']['destination']}\nPreferences: {state['plan'].get('traveler_preferences', [])}\nTravel style: {travel_style}{hotel_context}{weather_context}"
    )
    state["activities_notes"] = _parse_notes(output)
    return state


def places_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(PLACES_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')

    # ← ORCHESTRATOR PATTERN: Places receives hotel + activities context to avoid duplicates
    context_parts = ""
    if state.get('hotel_notes'):
        hotel_loc = chr(10).join('- ' + n for n in state['hotel_notes'][:3])
        context_parts += f"\n\n══ CONTEXT FROM HOTEL AGENT ══\nHotel location:\n{hotel_loc}\nRecommend places near the hotel. Include restaurants close to the hotel.\n══════════════════════════════\n"

    if state.get('activities_notes'):
        activities_summary = chr(10).join('- ' + n for n in state['activities_notes'][:5])
        context_parts += f"\n\n══ CONTEXT FROM ACTIVITIES AGENT ══\nActivities already recommended:\n{activities_summary}\nDo NOT duplicate these. Find DIFFERENT places, landmarks, and restaurants.\n══════════════════════════════════════\n"

    output = _invoke_agent(agent,
        f"Find best places, landmarks, restaurants for: {state['question']}\nDestination: {state['plan']['destination']}\nPreferences: {state['plan'].get('traveler_preferences', [])}\nTravel style: {travel_style}{context_parts}"
    )
    state["places_notes"] = _parse_notes(output)
    return state


def budget_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(BUDGET_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')
    num_days = state['plan']['num_days']

    # ── Extract num_travelers from the question ──
    num_travelers = 1
    for line in state['question'].split('\n'):
        line_stripped = line.strip().lower()
        if 'number of travelers' in line_stripped or 'عدد المسافرين' in line_stripped:
            # Extract the number after the colon
            if ':' in line_stripped:
                val = line_stripped.split(':', 1)[1].strip()
                try:
                    num_travelers = int(val)
                except ValueError:
                    pass
            break

    # Pass hotel and flight research so budget uses ACTUAL prices
    hotel_research = chr(10).join('- ' + n for n in state.get('hotel_notes', []))
    flight_research = chr(10).join('- ' + n for n in state.get('flight_notes', []))

    output = _invoke_agent(agent,
        f"Calculate travel budget for: {state['question']}\nDestination: {state['plan']['destination']}\nEXACT number of days: {num_days} (calculate for {num_days} days ONLY, not more)\nTravel style: {travel_style}\n\n"
        f"═══ CRITICAL: NUM_TRAVELERS = {num_travelers} ═══\n"
        f"Transport costs below are PER PERSON. You MUST multiply by {num_travelers}.\n"
        f"Example: if flight = 13,500/person, total transport = 13,500 × {num_travelers} = {13500 * num_travelers}\n"
        f"Activities priced per person: multiply by {num_travelers}.\n"
        f"Food costs: multiply by {num_travelers}.\n"
        f"Hotel: usually per ROOM, do NOT multiply unless explicitly per-person.\n"
        f"══════════════════════════════════════════════\n\n"
        f"══════ ACTUAL PRICES FROM OTHER AGENTS (use these, don't invent new ones) ══════\n"
        f"Hotel research (use the recommended hotel's actual price):\n{hotel_research}\n\n"
        f"Flight/Transport research (use these actual transport costs — these are PER PERSON):\n{flight_research}\n"
        f"══════════════════════════════════════════════════════════════════════════════════\n\n"
        f"IMPORTANT: Use the REAL prices above. The hotel cost in your budget MUST match the recommended hotel price.\n"
        f"The transport cost MUST = per_person_price × {num_travelers} travelers.\n"
        f"If flight agent says 13,500/person and there are {num_travelers} travelers, transport = {13500 * num_travelers}. NOT 13,500."
    )
    state["budget_notes"] = _parse_notes(output)
    return state


def coordinator_node(state: GraphState) -> GraphState:
    """Central Orchestrator — cross-validates ALL research and produces unified brief."""
    num_days = state['plan']['num_days']
    travel_style = state['plan'].get('travel_style', 'mid-range')
    destination = state['plan']['destination']

    # Gather ALL research notes
    flight_research = chr(10).join('- ' + n for n in state.get('flight_notes', []))
    hotel_research = chr(10).join('- ' + n for n in state.get('hotel_notes', []))
    visa_research = chr(10).join('- ' + n for n in state.get('visa_notes', []))
    weather_research = chr(10).join('- ' + n for n in state.get('weather_notes', []))
    activities_research = chr(10).join('- ' + n for n in state.get('activities_notes', []))
    places_research = chr(10).join('- ' + n for n in state.get('places_notes', []))
    budget_research = chr(10).join('- ' + n for n in state.get('budget_notes', []))

    resp = llm.invoke([
        SystemMessage(content=COORDINATOR_SYSTEM),
        HumanMessage(content=f"""User's original request:
{state['question']}

══════ PLAN METADATA ══════
Destination: {destination}
Dates: {state['plan']['travel_dates']}
Duration: {num_days} days
Travel Style: {travel_style}
Traveler Preferences: {state['plan'].get('traveler_preferences', [])}
Key Risks: {state['plan'].get('key_risks', [])}
════════════════════════════

══════ FLIGHT/TRANSPORT AGENT RESEARCH ══════
{flight_research or '(no data)'}

══════ HOTEL AGENT RESEARCH ══════
{hotel_research or '(no data)'}

══════ VISA AGENT RESEARCH ══════
{visa_research or '(no data)'}

══════ WEATHER AGENT RESEARCH ══════
{weather_research or '(no data)'}

══════ ACTIVITIES AGENT RESEARCH ══════
{activities_research or '(no data)'}

══════ PLACES AGENT RESEARCH ══════
{places_research or '(no data)'}

══════ BUDGET AGENT RESEARCH ══════
{budget_research or '(no data)'}

Now cross-validate ALL of the above data, resolve any conflicts,
and produce the UNIFIED BRIEF following your output format.
Make sure hotel price in budget matches actual hotel price.
Make sure transport cost in budget matches actual transport cost.
Make sure total never exceeds user's budget.
If this is domestic travel, remove any embassy/visa info."""),
    ]).content

    state["coordinator_brief"] = resp
    return state


def writer_node(state: GraphState) -> GraphState:
    headings = state["plan"].get("desired_output_structure", [])
    review_text = ""
    if state.get("review"):
        review_text = f"\n\nPrevious review to address:\n{json.dumps(state['review'], indent=2)}"

    num_days = state['plan']['num_days']
    travel_style = state['plan'].get('travel_style', 'mid-range')

    # ← ORCHESTRATOR PATTERN: Writer uses coordinator_brief as PRIMARY source
    coordinator_brief = state.get('coordinator_brief', '')

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

══════ CRITICAL CONSTRAINTS ══════
NUMBER OF DAYS: {num_days} — write EXACTLY {num_days} day(s) in the itinerary. Not {num_days + 1}, not {num_days - 1}.
TRAVEL STYLE: {travel_style} — ALL recommendations must match this style.
══════════════════════════════════

Plan: {json.dumps(state['plan'], indent=2)}
Output headings: {headings}

╔══════════════════════════════════════════════════════════════╗
║  COORDINATOR'S VALIDATED BRIEF — USE THIS AS PRIMARY SOURCE ║
║  All data below has been cross-validated for consistency.    ║
║  Hotel prices, transport costs, and budget are VERIFIED.     ║
╚══════════════════════════════════════════════════════════════╝

{coordinator_brief}

═══ RAW RESEARCH NOTES (for additional detail only) ═══

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

    # ← ORCHESTRATOR PATTERN: Finalizer uses coordinator_brief as PRIMARY source
    coordinator_brief = state.get('coordinator_brief', '')

    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

══════ CRITICAL CONSTRAINTS ══════
NUMBER OF DAYS: {num_days} — write EXACTLY {num_days} day(s). Count them before finishing.
TRAVEL STYLE: {travel_style} — every recommendation must match this style.
══════════════════════════════════

Plan: {json.dumps(state['plan'], indent=2)}

╔══════════════════════════════════════════════════════════════╗
║  COORDINATOR'S VALIDATED BRIEF — USE THIS AS PRIMARY SOURCE ║
║  All data below has been cross-validated for consistency.    ║
║  Hotel prices, transport costs, and budget are VERIFIED.     ║
╚══════════════════════════════════════════════════════════════╝

{coordinator_brief}

═══ RAW RESEARCH (for additional detail only) ═══
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
    transport = review.get("transport_realism", {})
    hotel_order = review.get("hotel_ordering", {})
    price_sel = review.get("price_selection", {})

    has_critical = (
        budget.get("is_over_budget", False)
        or budget.get("budget_underutilized", False)
        or not visa.get("visa_info_present", True)
        or visa.get("wrongly_marked_domestic", False)
        # ── New critical checks ──
        or not transport.get("transport_price_realistic", True)
        or not transport.get("price_matches_style", True)
        or not hotel_order.get("ordered_expensive_first", True)
        or not hotel_order.get("recommended_matches_style", True)
        or price_sel.get("flight_picked_cheapest_unnecessarily", False)
        or price_sel.get("hotel_picked_cheapest_unnecessarily", False)
        or not price_sel.get("style_appropriate_selections", True)
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
workflow.add_node("coordinator", coordinator_node)    # ← NEW: Central Orchestrator
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("finalizer", finalizer_node)

workflow.set_entry_point("planner")

# ── Hub-and-Spoke: Planner → Research Agents → Coordinator → Writer ──
workflow.add_edge("planner", "flight")
workflow.add_edge("flight", "hotel")          # hotel sees flight context
workflow.add_edge("hotel", "visa")
workflow.add_edge("visa", "weather")
workflow.add_edge("weather", "activities")    # activities see hotel + weather context
workflow.add_edge("activities", "places")     # places see hotel + activities context
workflow.add_edge("places", "budget")         # budget sees hotel + flight prices
workflow.add_edge("budget", "coordinator")    # ← ALL research → Coordinator
workflow.add_edge("coordinator", "writer")    # ← Coordinator's validated brief → Writer
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
        "coordinator_brief": None,
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
