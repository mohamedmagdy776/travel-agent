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
import re
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
    destination: str = Field(..., description="Travel destination (city/resort name)")
    destination_country: str = Field(..., description="Country where the destination is located (e.g. 'Egypt' for El Gouna, 'Saudi Arabia' for Jeddah)")
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
    user_budget: Optional[float]              # ← CODE-LEVEL: exact budget from form
    num_travelers: int                        # ← CODE-LEVEL: exact travelers from form
    user_currency: Optional[str]              # ← CODE-LEVEL: currency CODE from form (USD/EGP/…)


# ─────────────────────────────────────────────
# 4. System Prompts
# ─────────────────────────────────────────────

PLANNER_SYSTEM = """You are the Planner agent for a travel planning system.
Given a user's travel request:
1. Identify destination, dates, number of days, and traveler preferences.
2. Identify the COUNTRY where the destination is located (destination_country).
   Examples: El Gouna → "Egypt", شرم الشيخ → "Egypt", جدة → "Saudi Arabia", دبي → "UAE", بالي → "Indonesia"
3. Break planning into ordered steps.
4. Identify key risks (visa, weather, budget, safety).
5. Define output headings for the final itinerary.

CRITICAL — destination_country:
- This field identifies WHICH COUNTRY the destination city/resort is in.
- "الجونة" (El Gouna) is in Egypt → destination_country = "Egypt"
- "شرم الشيخ" (Sharm El Sheikh) is in Egypt → destination_country = "Egypt"
- "الغردقة" (Hurghada) is in Egypt → destination_country = "Egypt"
- "دبي" (Dubai) is in UAE → destination_country = "UAE"
- "جدة" (Jeddah) is in Saudi Arabia → destination_country = "Saudi Arabia"
- This is used to determine if the traveler needs a visa (different country = visa needed).

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

TRAVEL STYLE — HOW IT APPLIES TO TRANSPORT (READ CAREFULLY):
⚠️ THE GOLDEN RULE: the flight must NEVER break the budget. Within that rule, pick the BEST
   transport the budget can comfortably afford while still leaving enough for a great hotel +
   activities + dining.
   • If the budget COMFORTABLY covers a premium/business flight AND still leaves plenty for a
     high-level stay → a LUXURY traveler can absolutely have that premium/business flight. That is fine.
   • ONLY when the best flight would break the budget (or starve the hotel/activities) do you step
     DOWN — to the best economy/premium-economy, then to a cheaper transit/connecting route —
     because a flight is just transit, and the luxury is better spent on the stay than on an
     over-expensive ticket the budget can't really afford.
   In short: reduce the flight ONLY when it breaks the budget; otherwise the best affordable flight is good.

- LUXURY: Recommend the BEST flight the budget comfortably allows (business/premium is fine WHEN the
  budget covers it and the hotel + activities + dining are still well funded). If the best flight would
  break the budget, step down to the best economy within the transport cap and move the luxury into the stay.
  For ground legs, prefer a private transfer over a bus.
  If no direct flights exist, recommend: flight to nearest airport + private car/taxi transfer.
  Example: Dahab luxury = fly to Sharm El Sheikh + private transfer (1.5hrs, ~500-800 EGP).
- MID-RANGE: Recommend good-value economy on a reputable airline — comfort at a reasonable price.
- BUDGET: Recommend cheapest options — budget airlines, transit/connecting routes, shared transport.

ALWAYS also search the CHEAPEST routes (connecting / transit flights on carriers like Pegasus, AJet,
Aegean, Wizz, etc. are often far cheaper than a direct flagship-carrier flight) and include at least
one as an option — so that IF the budget is tight, there is a cheaper routing ready to fall back to.
Present options cheapest-to-most-expensive; recommend the best one that fits the budget comfortably.

Rules:
- Search for real transport options for the user's route and dates.
- Find best airlines, price ranges, flight duration, layovers.
- Include booking tips and best time to buy.
- DO NOT include any booking links — links will be added automatically by the system.
- If destination has no airport, suggest the BEST transport option for the travel style (see above).
- 🔎 ALWAYS use Tavily to search for the ACTUAL, CURRENT price of THIS exact route and dates
  BEFORE quoting anything. Do NOT answer from memory. Run at least 2-3 searches, e.g.:
    • "<from> to <destination> flight price <month year> round trip"
    • "<airline> <from> to <destination> ticket price <year>"
    • "cheapest flights <from> <destination> <dates>"
  Read the returned results and quote the REAL searched price (per person, round trip).
- The hardcoded ranges below are a SANITY CHECK ONLY — to catch obviously-wrong / hallucinated
  numbers, and a LAST-RESORT fallback when a search truly returns nothing. They are NOT a
  substitute for searching. A real searched price that falls inside the range ALWAYS wins over
  the middle-of-range estimate.

═══════════════════════════════════════════════════
PRICE REALISM — EXTREMELY CRITICAL (2025-2026 PRICES):
═══════════════════════════════════════════════════
- ALWAYS quote ROUND TRIP (return) prices per person, not one-way. State clearly "رايح وجاي" / "round trip".
- Egyptian domestic flight prices have INCREASED significantly. The prices below are MINIMUM realistic
  ranges for 2025-2026. If your Tavily search returns prices BELOW these minimums, the data is WRONG
  or outdated — use the ranges below instead.

DOMESTIC FLIGHTS WITHIN EGYPT (round trip per person, economy):
  Cairo → Sharm El Sheikh: 6,000 - 13,000 EGP (budget airlines ~6,000, EgyptAir ~10,000-13,000)
  Cairo → Hurghada:        6,000 - 12,000 EGP (budget airlines ~6,000, EgyptAir ~9,000-12,000)
  Cairo → Luxor:           5,000 - 10,000 EGP
  Cairo → Aswan:           6,000 - 11,000 EGP
  Cairo → Marsa Alam:      7,000 - 13,000 EGP
  ABSOLUTE MINIMUM for ANY domestic flight in Egypt: 5,000 EGP round trip per person.
  If you find a price under 5,000 EGP, it is WRONG — do NOT use it.

BUSINESS CLASS (domestic Egypt, round trip per person):
  Typically 1.5x-2.5x economy price: 10,000 - 25,000 EGP depending on route.

INTERNATIONAL FLIGHTS FROM EGYPT (round trip per person, economy):
  Egypt → Gulf countries (UAE, Saudi, Kuwait, Qatar): 12,000 - 25,000 EGP ($250 - $500 USD)
  Egypt → Turkey: 10,000 - 20,000 EGP ($200 - $400 USD)
  Egypt → Europe (Italy, France, UK, Spain, Germany): 15,000 - 35,000 EGP ($300 - $700 USD)
  Egypt → Southeast Asia: 20,000 - 45,000 EGP ($400 - $900 USD)
  ABSOLUTE MINIMUM for ANY international flight: 10,000 EGP / $200 USD round trip per person.

INTERNATIONAL FLIGHTS IN USD (round trip per person, economy — 2025-2026):
  ══════════════════════════════════════════════════════════════
  Cairo → Rome/Milan (Italy): $250 - $500 USD (EgyptAir ~$300, budget airlines ~$200-250)
  Cairo → Istanbul (Turkey): $200 - $400 USD
  Cairo → Paris/London/Berlin: $300 - $600 USD
  Cairo → Zurich/Geneva (Switzerland): $350 - $600 USD economy direct (SWISS/EgyptAir ~$400-600);
    transit routes (via Istanbul/Athens on Pegasus, AJet, Aegean) are often $230 - $400 USD round trip.
  Cairo → Dubai/Abu Dhabi: $200 - $400 USD
  Cairo → Bangkok/KL: $400 - $800 USD
  ANY Middle East → Europe: $300 - $700 USD economy (transit routes cheaper, $230-450)
  ANY Middle East → SE Asia: $400 - $900 USD
  USA/Canada → Europe: $400 - $900 USD
  ABSOLUTE MAXIMUM for ECONOMY round trip: $900 USD (an ECONOMY ticket above this = WRONG data).
  ⚠️ A price of $1,200-1,600 is NOT economy — it is BUSINESS/premium class (or a hallucinated economy price).
     So do NOT label such a price as "economy". Cairo→Switzerland ECONOMY is $350-600, NEVER $1,500+ as economy.
     (You MAY still quote a genuine business/premium flight at that price — but only when the budget
      comfortably affords it; and always label it clearly as business/premium, not economy.)
  ══════════════════════════════════════════════════════════════
  ⚠️ If you quote a price ABOVE these ranges, you are HALLUCINATING.
  A Cairo→Rome economy round trip is $250-$500, NEVER $1,000+.
  If Tavily returns no price, use the MIDDLE of these ranges.

INTERNATIONAL FLIGHTS TO EGYPT (round trip per person, economy):
  Kuwait → Egypt (Cairo/Hurghada/Sharm): 8,000 - 18,000 EGP / $160 - $360 USD
  Saudi → Egypt: 8,000 - 18,000 EGP / $160 - $360 USD
  UAE → Egypt: 10,000 - 20,000 EGP / $200 - $400 USD

GROUND TRANSPORT (one-way prices):
  Private car/transfer Cairo→Hurghada (5-6 hours): 4,000-7,000 EGP
  Private car/transfer Cairo→Sharm (via Suez): 5,000-8,000 EGP
  Private car/transfer Sharm→Dahab (1.5hrs): 1,500-3,000 EGP
  Private car/transfer airport to hotel (short): 500-1,500 EGP
  Bus Cairo→Hurghada: 400-800 EGP per person
  Bus Cairo→Sharm: 400-800 EGP per person
  Bus Cairo→Dahab: 500-900 EGP per person

CRITICAL VALIDATION:
- A flight ticket (one-way) is NEVER under 2,500 EGP. Under 2,500 = WRONG DATA.
- A round-trip flight is NEVER under 5,000 EGP. Under 5,000 = WRONG DATA.
- If Tavily returns a price like 2,000 EGP for a round-trip flight, that is OLD/WRONG data.
  IGNORE it and use the realistic ranges above.
- ALWAYS search Tavily for current prices BUT validate against the ranges above.
  If Tavily price < minimum range → use the MINIMUM of the range.
  If Tavily price is within range → use the Tavily price.
  If Tavily returns no price → use the MIDDLE of the range.
- When in doubt, quote the HIGHER end of the realistic range, not the lower end.
- These ranges already account for 2025-2026 price inflation in Egypt.

═══════════════════════════════════════════════════
PRICE SELECTION BY TRAVEL STYLE — BUDGET-AWARE:
═══════════════════════════════════════════════════
⚠️ CRITICAL RULE: Transport should NEVER exceed 40% of the TOTAL TRIP BUDGET.
The user's total budget must cover: transport + hotel + activities + food + misc.
If transport alone takes 50%+ of the budget, the plan WILL fail budget validation.

STEP 1 — CHECK BUDGET FEASIBILITY:
- Look at the user's total budget and number of days.
- Estimate: hotel needs ~40-50% of budget, transport ~20-30%, activities ~15-20%, food ~10%.
- If the best flight option would exceed 30-40% of budget → recommend a cheaper option.
- Example: Budget = 12,000 AED, 4 days, 2 people.
  Max transport = 12,000 × 30% = 3,600. So max flight/person = 1,800.
  Business class at 6,000/person = 12,000 total = 100% of budget → IMPOSSIBLE.
  Economy at 2,000/person = 4,000 total = 33% → FITS.
  Recommend ECONOMY even for luxury traveler because business class would blow the budget.

STEP 2 — STYLE-AWARE SELECTION (within budget):
- LUXURY travelers: Recommend the BEST flight the budget comfortably allows. If the budget covers a
  premium/business flight AND still leaves the hotel + activities + dining well funded → recommend that
  premium/business flight; that is perfectly fine. ONLY if the best flight would exceed the transport
  cap (or starve the stay) do you step down to the best economy within the cap and move the luxury
  into the hotel/experiences. Reduce the flight ONLY when it would break the budget.
- MID-RANGE travelers: Recommend good-value economy on a decent airline — comfort at a reasonable price.
- BUDGET travelers: Recommend the cheapest practical option (budget carriers, transit routes).
- NEVER recommend an option that alone exceeds 40% of the total budget, regardless of style.
- ALWAYS also surface at least one cheaper routing (transit/connecting carriers) as a fallback option,
  so that IF the budget is tight there is a cheaper way ready — but you do NOT have to pick it when the
  budget comfortably affords something better.
- If ALL options exceed 40% of budget, recommend the cheapest one and add a NOTE:
  "⚠️ تكلفة الطيران مرتفعة مقارنة بالميزانية. يُنصح بزيادة الميزانية أو اختيار وسيلة نقل بديلة."

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

🔎 ALWAYS SEARCH FIRST — MANDATORY:
Before quoting ANY hotel or price, use Tavily to find REAL, currently-bookable hotels in the exact
destination and their ACTUAL nightly prices for the travel dates. Do NOT answer from memory. Run at
least 2-3 searches, e.g. "best 5-star hotels <destination> price per night <month year>",
"<destination> all inclusive resort price <year>", "<hotel name> price per night". Quote the REAL
searched price. The ranges below are a SANITY CHECK + last-resort fallback only — never a substitute
for searching, and a real searched price inside the range always wins over the middle estimate.

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

   REALISTIC HOTEL PRICES IN USD (2025-2026) — USE THESE AS ANCHORS:
   ══════════════════════════════════════════════════════════════
   EUROPE (Rome, Paris, Istanbul, Barcelona, London):
     Budget (3-star): $80 - $150/night
     Mid-range (4-star): $150 - $300/night
     Luxury (5-star): $250 - $800/night
     Ultra-luxury (palace hotels): $800 - $2,000/night
   MIDDLE EAST (Dubai, Abu Dhabi, Doha):
     Budget: $60 - $120/night
     Mid-range: $120 - $250/night
     Luxury: $200 - $600/night
   SOUTHEAST ASIA (Bangkok, Bali, KL):
     Budget: $30 - $80/night
     Mid-range: $80 - $200/night
     Luxury: $150 - $500/night
   EGYPT (Sharm, Hurghada, Cairo, Luxor) in EGP:
     Budget: 1,500 - 3,500 EGP/night
     Mid-range: 3,500 - 8,000 EGP/night
     Luxury: 5,000 - 15,000 EGP/night
     All-inclusive resort: 4,000 - 20,000 EGP/night
   ══════════════════════════════════════════════════════════════
   ⚠️ If you quote a price ABOVE these ranges, you are HALLUCINATING.
   A 5-star hotel in Rome is $250-$800/night, NEVER $1,500/night for a standard room.
   If Tavily returns no price, use the MIDDLE of these ranges.

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

6. RECOMMENDATION LOGIC — BUDGET IS THE HARD LIMIT:
   ⚠️ BUDGET COMES FIRST, STYLE COMES SECOND.
   - The user's budget LIMIT is NON-NEGOTIABLE. NEVER recommend a hotel that exceeds max_per_night.
   - LUXURY travelers: recommend the BEST hotel WITHIN the budget limit.
     If max_per_night is $300 and the best 5-star is $800, DO NOT recommend it.
     Instead recommend the best 4-star at $250-300. That IS luxury within this budget.
     Add a note: "لو عايز فندق 5 نجوم فاخر، يُنصح بزيادة الميزانية إلى [amount]."
   - MID-RANGE travelers: recommend best quality/price balance within budget.
   - BUDGET travelers: recommend best value for money.
   - If the budget only allows a 3-star hotel, recommend the BEST 3-star, not a 5-star that blows the budget.
   - Think: "What is the BEST experience this budget can buy?" not "What is the most expensive hotel?"

- Use Tavily to search and verify real hotels and their ACTUAL prices in the destination.
- Show budget calculation before the 3 options.
- IMPORTANT: The hotel you recommend here will appear in ALL sections of the plan.
  Make sure your recommendation matches what a traveler of this style actually wants."""

VISA_SYSTEM = """You are the Visa agent. You have access to a web search tool called Tavily.

═══════════════════════════════════════════════════
⛔ SCOPE — WHAT YOU OUTPUT (AND WHAT YOU NEVER OUTPUT):
═══════════════════════════════════════════════════
You output ONLY visa / passport / border-entry information. NOTHING ELSE.

NEVER include in your output — these belong to OTHER agents and must NOT appear here:
  ❌ Hotels, resorts, accommodation, room prices, "الفنادق" — handled by the Hotel agent.
  ❌ Activities, excursions, tours, diving, "الأنشطة" — handled by the Activities agent.
  ❌ Restaurants, cafes, dining, food — handled by the Places agent.
  ❌ Attractions, beaches, sights, places to visit — handled by the Places agent.
  ❌ Weather, temperature, what to pack — handled by the Weather agent.
  ❌ A general "travel guide" dump of the destination.

If your Tavily search returns a travel-guide page that also lists hotels, activities,
or attractions, IGNORE all of that. Extract ONLY the visa/entry/passport facts.
Your entire answer must be visa/passport/entry requirements and nothing else.

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
- Flights (round trip): ~6,000-10,000 EGP
- Activities: ~3,000-5,000 EGP
- Transport (airport transfer): ~800-1,500 EGP
- Misc: ~2,000 EGP
- TOTAL: ~20,000-33,000 EGP → Remaining: 27,000-40,000 EGP
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
- If remaining > 0 for LUXURY / MID-RANGE: the plan is UNDER-USING the budget — it must be CONSUMED
  with real, named experiences (repeating a great one is fine), not reported as spare money. Do NOT
  frame leftover as "can be used for...". Only for short trips / budget travelers, say
  "الميزانية المتبقية: X" plainly (remaining budget: X), with no suggestion.
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
- Use Tavily to verify current prices for any cost NOT already provided by the flight/hotel agents
  (e.g. activity tickets, meals, airport transfers). Prefer real searched prices over estimates.
- After searching, formulate specific bullet-point notes with the budget math shown clearly."""

WRITER_SYSTEM = """You are the Writer agent for a travel planning system.
Write a detailed, specific, useful travel plan.

═══════════════════════════════════════════════════
DAY COUNT — THE MOST CRITICAL RULE:
═══════════════════════════════════════════════════
- The CONSTRAINTS section of each request gives you the EXACT "ITINERARY DAYS" number
  and the "HOTEL NIGHTS" number. Use those EXACT numbers — do not invent your own.
- The itinerary spans from the ARRIVAL day through the CHECK-OUT day:
    • Day 1 = arrival day (check-in ~noon; afternoon/evening activities only).
    • Middle days = full days.
    • The LAST day = check-out / departure day (morning only; check-out + airport transfer).
- Because arrival and check-out are separate days, ITINERARY DAYS = HOTEL NIGHTS + 1.
  Example: 5 nights booked → 6 dated days (Day 1 arrival … Day 6 check-out).
- Write EXACTLY the "ITINERARY DAYS" number of dated days. No more, no less.
- NEVER merge the check-out into the last full day — check-out is its own final day.
- VIOLATING THE DAY COUNT IS THE WORST POSSIBLE ERROR.

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
| الخيار 1 (الأفضل) | Xhr | X ج.م | X × عدد = Y ج.م | رحلة رايح وجاي، اقتصادي |
| الخيار 2 | Xhr | X ج.م | X × عدد = Y ج.م | رايح وجاي، اقتصادي شركة أخرى |
| الخيار 3 (الأرخص) | Xhr | X ج.م | X × عدد = Y ج.م | رايح وجاي |
CRITICAL RULES:
- ALWAYS show EXACTLY 3 transport options. No more, no less. If flight research has fewer, add bus/private car options.
- ALWAYS show price PER PERSON and TOTAL for all travelers.
- If 3 travelers and flight costs 8,000/person, write "8,000 ج.م" under التكلفة/فرد and "24,000 ج.م" under الإجمالي.
- ALWAYS specify if the price is round trip (رايح وجاي) or one-way.
- Show ALL 3 transport options, ordered from most expensive/comfortable to cheapest.
- ⚠️ BUDGET CHECK: If ANY option's الإجمالي exceeds 35% of user's total budget, DO NOT recommend it.
  Instead, recommend the most expensive option that stays under 35% of budget.
- The AI recommendation must pick an option where الإجمالي ≤ 35% of total budget.
- Write: "🏆 توصية الـ AI: [الخيار] — لأن [السبب]"
IMPORTANT: Do NOT repeat transport options anywhere else.

═══════════════════════════════════════════════════
TRANSPORT TABLE — INCLUDE ALL LEGS:
═══════════════════════════════════════════════════
- Each transport option's "التكلفة/فرد" MUST include ALL transport legs for that option.
- Example: "الطيران إلى شرم الشيخ + نقل خاص إلى دهب" means:
  Flight Cairo→Sharm = 6,500/person + Private car Sharm→Dahab = 800/person
  التكلفة/فرد = 7,300 ج.م (NOT just 6,500)
  الإجمالي = 7,300 × 4 = 29,200 ج.م
- NEVER split legs across rows. One option = one row = ALL costs combined.
- The "الإجمالي" in the transport table is the FINAL transport number for that option.
  This EXACT number goes into the budget. No additions, no separate line items later.

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
- If total > budget: ERROR — reduce costs.
- REMAINING BUDGET — DO NOT just report it as "spare money":
  • For LUXURY / MID-RANGE trips (any length): you MUST first CONSUME the remainder with real,
    named additions (see the next section) until it is nearly gone (remaining ≈ 0-3% of budget).
  • ⛔ NEVER write a suggestion note like "الميزانية المتبقية: X، يمكن استخدامها لتحسين الأنشطة أو
    تناول الطعام في مطاعم فاخرة". That phrasing is FORBIDDEN — YOU are the planner: actually ADD those
    meals / activities to the day-by-day itinerary and the budget table, instead of telling the user
    they "could" spend it.
  • Only if a small residual is genuinely unavoidable, state it plainly: "الميزانية المتبقية: X" —
    with NO "يمكن استخدامها" / "can be used for" suggestion attached.
  • If remaining = 0: write "تم استخدام الميزانية بالكامل". NEVER say "تم استهلاك الميزانية".
- SHORT TRIPS (1-3 days) & BUDGET-style travelers are the exception: a large remaining balance is
  normal/good — just report the real remaining amount plainly, with no suggestion text.

═══════════════════════════════════════════════════
USE THE BUDGET — FILL LEFTOVER WITH EXPERIENCES (multi-day non-budget trips):
═══════════════════════════════════════════════════
The user set a budget to be USED, not hoarded. The goal is: "I gave you a budget — use it up on a
great trip." So for LUXURY and MID-RANGE trips of 4+ days, do NOT leave a large remaining balance.
- AFTER you've picked the hotel and transport, check how much budget is left.
- If the remaining is large (roughly >10% of the total budget), FILL IT with real, named experiences
  — do NOT just inflate hotel/flight prices. Add value the traveler actually enjoys:
    1. More dining experiences — name specific premium/notable restaurants, with a price per meal.
    2. Extra excursions / day-trips / activities — name them (operator, location, price/person).
    3. Outings and add-ons — spa, private boat, sunset cruise, guided tours, premium entry tickets.
- Distribute these ACROSS the daily itinerary (e.g. a nice dinner each evening, one excursion per
  full day) so the days feel full, and ADD their costs to the budget table (Activities / Dining rows).
- Keep adding until the remaining balance is NEARLY ZERO — target using ~97-100% of the budget. Only
  a tiny residual (≤3% of the budget) may remain.
- ✅ It is PERFECTLY FINE to REPEAT a standout experience to use up the budget — e.g. book a second
  dinner at the same great restaurant, add another day of a loved activity, or a second excursion.
  Listing a real, named place twice (with its price counted twice) is better than leaving money unspent.
- HARD RULE: the total must still be LESS than the budget. Never go over. Fill UP TO it, never past it.
- BUDGET-style travelers are the exception: for them, a large remaining balance is GOOD (they save).
- Every added restaurant/activity must be REAL and NAMED with a price — never "extra dining ~5,000".

═══════════════════════════════════════════════════
FOOD / DINING IN THE BUDGET — MUST BE COUNTED (CRITICAL):
═══════════════════════════════════════════════════
⛔ "Food: 0" is FORBIDDEN whenever the itinerary lists ANY paid meal (lunch, dinner, restaurant).
   A budget that shows Michelin dinners in the day-by-day plan but "Food: 0" in the table is WRONG
   and self-contradictory — it will be rejected.
- SUM every paid meal you put in the itinerary and put that sum in the "الطعام / Food" row of the budget
  table. The Food row MUST equal the total of the meals actually scheduled (per-person meals × travelers,
  group meals as-is). The itinerary and the Food row MUST agree to the dollar.
- Food is the ONLY exception when it's genuinely 0: the hotel meal plan is Full-Board / All-Inclusive
  AND no extra paid restaurant appears in the itinerary. In that case write "0 (مشمول في الفندق)".
- KEEP THE RESTAURANTS LUXURIOUS, BUT LIMIT HOW MANY expensive outings you schedule so the food total
  fits the budget. Do NOT put a $150/person Michelin dinner every single night. Instead:
    • Feature 2–3 signature premium/Michelin dinners across the whole trip (the highlights).
    • On the other evenings, keep it lighter: a nice but moderate local restaurant, or a meal included
      in the hotel. Quality over quantity — fewer grand dinners, each one special.
- The TOTAL (transport + hotel + activities + FOOD + misc) must include food and stay UNDER the budget,
  while still using ~90–97% of it. If adding the real food cost pushes you over, REDUCE THE NUMBER of
  premium dinners (keep them luxurious, just fewer) — do NOT hide the food cost to look within budget.

═══════════════════════════════════════════════════
TRANSPORT IN BUDGET — MULTIPLY BY TRAVELERS:
═══════════════════════════════════════════════════
- The transport table already shows TOTAL for all travelers in the "الإجمالي" column.
- The budget transport number MUST be EXACTLY the "الإجمالي" from the AI-recommended option.
- Do NOT recalculate or add separate line items. The transport table total IS the budget transport.
- Example: If AI recommends Option 1 and its الإجمالي = 29,200 → budget transport = 29,200.
- If these numbers don't match, the plan is WRONG. Fix it before writing.

═══════════════════════════════════════════════════
CONSISTENCY — EXTREMELY CRITICAL:
═══════════════════════════════════════════════════
⚠️ THE USER SEES TWO TABS: "البرنامج الكامل" AND "الميزانية".
⚠️ BOTH TABS MUST SHOW THE EXACT SAME NUMBERS. ANY DIFFERENCE IS A BUG.

The budget section you write in the draft will appear on Tab 1.
The budget_notes from the Budget agent will appear on Tab 4 (الميزانية).
If YOUR numbers differ from budget_notes, the user sees conflicting data on two tabs.

RULES:
1. The hotel in your budget MUST be the AI-recommended hotel from the hotels table.
   ⚠️ MANDATORY SELF-CHECK before finishing: take the per-night price of the hotel you marked
   🏆 توصية الـAI, multiply by the number of nights (× number of rooms if the group needs more than
   one). The الإقامة/الفندق line in the budget MUST equal exactly that. Example: recommended =
   22,000/night, 5 nights → الإقامة = 110,000 (NOT 60,000 from a cheaper hotel). NEVER let the budget
   quietly use a cheaper hotel than the one you recommended.
2. The hotel price/night MUST match what's in the hotels table AND in budget_notes.
3. The transport cost in budget MUST be the "الإجمالي" from the AI-recommended transport option.
4. Activity costs, food costs MUST match budget_notes.
5. The TOTAL and REMAINING budget MUST match budget_notes.
6. Do NOT invent your own numbers. Copy the exact figures from budget_notes.
7. If budget_notes say transport = 29,200 but your transport table says 32,000, then
   FIX the transport table to match, or vice versa. They MUST be identical.
8. If TOTAL > user budget: DO NOT just show negative remaining. FIX IT:
   - Switch to cheaper transport/hotel options that fit within budget.
   - If nothing fits, use cheapest options and add:
     "⚠️ الميزانية لا تكفي لتغطية كل التكاليف. يُنصح بزيادة الميزانية."
   - NEVER produce a plan with negative remaining without explaining why.
9. Transport should NEVER exceed 40% of total budget. If it does, pick a cheaper option.

PROCESS: Write the transport table first → note the الإجمالي for the recommended option →
use THAT exact number in the budget section. Then cross-check against budget_notes.

DO NOT include a "نظرة عامة على الرحلة" (trip overview) section. This information is already
displayed in the app header (destination, dates, travelers, budget). Starting the plan with
an overview repeats what the user already sees. Start directly with the hotels/transport tables.

Use ALL research notes: flights/transport, hotels, visa, weather, activities, places, budget.
If a review exists, incorporate the fixes."""

REVIEWER_SYSTEM = """You are the Travel Plan Reviewer — a domain-specific validator.
You validate the LOGIC and COMPLETENESS of the travel plan.

## DAY COUNT validation — MOST CRITICAL
- The CONSTRAINTS give an EXPECTED ITINERARY DAYS number (= nights booked + 1, because the
  arrival day and the check-out day are separate dated days). Count the dated days in the itinerary.
- The itinerary MUST run Day 1 = arrival/check-in … LAST day = check-out/departure.
- If the dated-day count is MORE than expected, this is a CRITICAL error (-25 points).
- If it is FEWER than expected (e.g. check-out was crammed into the last full day instead of
  being its own day), this is also an error (-15 points).
- Add to fix_instructions: "Day count mismatch: expected X dated days (N nights + arrival + check-out)
  but itinerary has Y. Fix to exactly X days, with the last day being a separate check-out day."

## Travel Style validation
- Check if recommendations match the travel style (luxury/mid-range/budget).
- If travel_style is "luxury" but plan recommends buses or budget hotels: -15 points.
- Add to fix_instructions: "Travel style mismatch: user chose luxury but plan recommends [budget option]. Replace with premium alternative."

## Budget validation — CRITICAL
- Extract user's budget vs estimated total. Flag if over budget.
- If TOTAL COST > USER BUDGET: this is a CRITICAL error. Score MUST be < 80.
  -30 points. The plan MUST be revised to fit within budget.
  fix_instructions: "OVER BUDGET by [amount]. Total [total] exceeds budget [budget].
  Reduce costs: pick cheaper transport, cheaper hotel, or fewer activities.
  Transport should NOT exceed 40% of total budget."
- If transport alone > 40% of budget: -20 points.
  fix_instructions: "Transport takes [X]% of budget — too high. Pick cheaper transport option."
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

## LOGICAL CONTRADICTIONS — READ THE WHOLE PLAN AND CATCH THE OBVIOUS ONES (score < 75 if found):
Cross-check the budget table against the day-by-day itinerary. These are contradictions a careful
reader spots instantly — if you find one it is a FAIL: set the score below 75, set has_critical TRUE,
and add a clear Arabic fix instruction telling the writer to go back and fix it.
- FOOD vs RESTAURANTS: if the budget shows الطعام / Food = 0 (or "مشمول في الفندق") BUT the itinerary
  schedules ANY paid outside restaurant/meal with a price ("مطعم ... (300 ج.م للشخص)"), that is a
  CONTRADICTION. The food row MUST equal the sum of those paid meals × travelers. fix_instructions:
  "الطعام مكتوب 0 بينما الخطة فيها مطاعم مدفوعة — احسب الطعام من المطاعم المذكورة وحدّث الإجمالي والمتبقي."
- Any budget number that disagrees with the same thing elsewhere in the plan (hotel / transport /
  meals / activities) is a contradiction — send it back.
- Money left unused while an item is shown as 0 / free, or a plan that looks thin for the budget → send back.

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

## Transport price realism validation (2025-2026 prices)
- Domestic flights in Egypt: minimum 5,000 EGP round trip per person. NEVER under 5,000 EGP.
  Cairo → Sharm: 6,000-13,000 EGP | Cairo → Hurghada: 6,000-12,000 EGP
  Cairo → Luxor: 5,000-10,000 EGP | Cairo → Aswan: 6,000-11,000 EGP
- If flight price is under 5,000 EGP per person (e.g., 2,000 EGP), flag as UNREALISTIC: -25 points.
- International flights from Egypt: minimum 10,000 EGP (Gulf/Turkey), 15,000 EGP (Europe).
- Private car transfers 5-6 hours: 4,000-8,000 EGP one way. Under 3,000 EGP is UNREALISTIC.
- Bus tickets: 400-800 EGP per person one way. Under 200 EGP is UNREALISTIC.
- All transport prices should be ROUND TRIP unless clearly stated as one-way.

## Hotel ordering validation
- Hotels MUST be ordered from most expensive to cheapest (Option 1 = most expensive).
- If cheapest hotel is listed first, flag: -10 points.
- For LUXURY travelers: the recommended hotel MUST be the most expensive option. Flag if not: -15 points.

## Domestic travel validation
- If traveler is traveling within their own country, there should be NO embassy info. Flag if present: -10 points.

## STRUCTURED OUTPUT — YOU MUST FILL THESE FIELDS:

### transport_realism (TransportRealismCheck):
- transport_price_realistic: Is the transport price within the realistic range? (min 5,000 EGP domestic flight)
- reported_price: The actual transport price shown in the plan.
- realistic_min: Minimum realistic price for this route (e.g., 6,000 for Cairo→Hurghada).
- realistic_max: Maximum realistic price for this route (e.g., 12,000 for Cairo→Hurghada).
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
- BUDGET EXCEEDED (total > user budget): -30 (CRITICAL — forces score < 80)
- Transport > 40% of budget: -20 | Missing visa: -15 | Unrealistic schedule: -10
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
- Transport table total ≠ budget transport cost: -25
- Transport option has split legs not combined in total: -15

## Budget-Transport consistency validation
- The "الإجمالي" in the transport table for the recommended option MUST equal the transport cost in the budget.
- If transport table says 8,000 but budget says 11,200 → INCONSISTENCY: -25 points.
  fix_instructions: "Transport table total and budget transport cost don't match. Make them identical."
- Check that the transport table's per-person price INCLUDES all legs (flight + transfer + any other transport).
  If the option says "طيران + نقل خاص" but the per-person price only covers the flight → -15 points.
  fix_instructions: "Transport per-person price must include ALL legs. Add transfer cost to per-person price."

CRITICAL: If budget_check.is_over_budget is True → score MUST be < 80. ALWAYS.
  An over-budget plan is NEVER acceptable. It MUST go back to the Writer for cost reduction.
CRITICAL: If transport cost > 40% of total budget → score MUST be < 80.
CRITICAL: If transport_realism.transport_price_realistic is False → score MUST be < 80.
CRITICAL: If hotel_ordering.recommended_matches_style is False for luxury → score MUST be < 80.
CRITICAL: If price_selection.flight_picked_cheapest_unnecessarily is True for luxury → score MUST be < 80.
CRITICAL: If budget_check.budget_underutilized is True for luxury → score MUST be < 80.
These ensure the plan gets sent BACK to the Writer for fixes.

⚠️ OVER-BUDGET IS THE #1 MOST CRITICAL ERROR. No plan should EVER pass review if total > budget.
If the budget is genuinely too small for the trip, the plan should:
1. Use the cheapest viable options for everything
2. Show exactly what the money covers
3. Add a note: "الميزانية لا تكفي لتغطية الرحلة بالكامل. يُنصح بزيادة الميزانية إلى [suggested amount]."
But the plan should NEVER show negative remaining budget without the Reviewer catching it.

Return JSON matching the ReviewResult schema."""

FINALIZER_SYSTEM = """You are the Finalizer agent.
Produce the ultimate polished travel itinerary from all research and drafts.

═══════════════════════════════════════════════════
DAY COUNT — ABSOLUTE RULE:
═══════════════════════════════════════════════════
- The CONSTRAINTS give the EXACT "ITINERARY DAYS" and "HOTEL NIGHTS" numbers. Use them.
- The itinerary spans arrival → check-out, so ITINERARY DAYS = HOTEL NIGHTS + 1:
    • Day 1 = arrival day (check-in ~noon; afternoon/evening only).
    • Middle days = full days.
    • The LAST day = check-out / departure day (morning only; check-out + airport transfer).
- Example: 5 nights → 6 dated days (Day 1 arrival … Day 6 check-out).
- Count the dated days in your output before finishing. If it doesn't match ITINERARY DAYS, FIX IT.
- NEVER merge check-out into the last full day — check-out is its own separate final day.

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

4. Day-by-day itinerary with times, places, costs (EXACTLY the ITINERARY DAYS from the
   constraints — arrival day through a separate check-out day)
5. Practical info (transport, money, language, safety)
6. Packing checklist (based on weather)
7. Emergency info (hospital, police numbers)
   IMPORTANT: Do NOT include embassy info for DOMESTIC travel.
   If the traveler is traveling within their own country (e.g., Egyptian in Egypt), NO embassy needed.
   Only include embassy for international travel.
8. Budget breakdown at the END (total by category vs user budget)

═══════════════════════════════════════════════════
USE THE BUDGET — FILL LEFTOVER WITH EXPERIENCES (LUXURY / MID-RANGE, 4+ days):
═══════════════════════════════════════════════════
The user gave a budget to be USED on a great trip, not left sitting unused.
- After the hotel + transport are set, if a large balance remains (roughly >10% of the budget),
  FILL it with REAL, NAMED experiences — NOT by inflating hotel/flight prices:
    • more dining (named premium restaurants, price/meal),
    • extra excursions / day-trips (named operator + location + price/person),
    • outings & add-ons (spa, private boat, sunset cruise, guided tours, premium tickets).
- Spread them across the daily itinerary so the days feel full, and add their costs to the
  budget table under Activities / Dining.
- Target using ~97-100% of the budget. Only a tiny residual (≤3%) may remain; a bigger unused
  balance is NOT acceptable.
- ✅ It is FINE to REPEAT a standout experience to use up the budget — a second dinner at the same
  great restaurant, another day of a loved activity, a second excursion. Counting a real, named place
  twice is better than leaving money unspent.
- HARD RULE: total must stay BELOW the budget — fill UP TO it, never past it.
- BUDGET-style travelers are the exception: for them a large remaining balance is GOOD.

═══════════════════════════════════════════════════
FOOD / DINING — MUST BE COUNTED, NEVER 0 WITH RESTAURANTS (CRITICAL):
═══════════════════════════════════════════════════
⛔ "Food: 0" is FORBIDDEN whenever the day-by-day plan lists any paid meal. A plan that shows
   $150/person Michelin dinners but "Food: 0" in the budget table is self-contradictory and REJECTED.
- The "الطعام / Food" row MUST equal the SUM of every paid meal in the itinerary
  (per-person meals × travelers, group meals as-is). The table and the itinerary MUST agree.
- KEEP the restaurants luxurious, but LIMIT how many expensive outings are scheduled so food fits:
  feature 2–3 signature premium/Michelin dinners across the trip; make the other evenings lighter
  (a moderate local restaurant, or a meal included in the hotel). Fewer grand dinners, each special.
- If counting the real food cost pushes the TOTAL over budget, REDUCE THE NUMBER of premium dinners
  (keep them luxurious, just fewer) — NEVER hide food cost as 0 to appear within budget.
- Food = 0 is allowed ONLY when the hotel plan is Full-Board/All-Inclusive AND no paid restaurant
  appears in the plan; then write "0 (مشمول في الفندق)".

If a review exists, incorporate ALL fixes.
Make it ready to print and follow.

BUDGET PLACEMENT:
- The complete budget summary goes at the END under "💰 الميزانية" or "💰 Budget".
- Transport section shows only transport options and prices.
- NEVER put the full budget breakdown inside the transport/flights section.

BUDGET MATH — MUST BE CORRECT:
- Add costs exactly. Remaining = budget - total.
- If remaining > 0 for a LUXURY / MID-RANGE trip: do NOT leave it — CONSUME it with real, named meals
  / activities (repeating a great one is fine) until remaining ≈ 0-3%. ⛔ NEVER write a suggestion like
  "الميزانية المتبقية: X، يمكن استخدامها لـ..." — add the experiences yourself instead. If a tiny
  residual is unavoidable, state "الميزانية المتبقية: X" plainly, with NO "يمكن استخدامها" suggestion.
- If remaining = 0: write "تم استخدام الميزانية بالكامل". NEVER say "تم استهلاك الميزانية".
- For short trips (1-3 days) & BUDGET travelers, total will be MUCH less than budget. This is normal
  and correct — just report the real remaining amount plainly.
- The hotel in budget MUST match the recommended hotel in the hotels section.
  ⚠️ SELF-CHECK: الإقامة/الفندق line = (price/night of the 🏆 recommended hotel) × nights (× rooms if
  the group needs more than one). Example: 22,000/night × 5 nights = 110,000, NOT 60,000 from a cheaper
  hotel. Never let the budget quietly use a cheaper hotel than the recommended one.
- Transport cost in budget MUST be the TOTAL for ALL travelers (per_person × num_travelers).
  If transport table shows 13,500/person × 4 travelers = 54,000, budget transport = 54,000.
  NOT 13,500. The budget transport MUST match the "الإجمالي" column in the transport table.

CONSISTENCY CHECK BEFORE FINALIZING:
- Hotel name in hotels table = hotel in budget = hotel in itinerary (all same)
- Transport table per-person price INCLUDES all transport legs (flight + transfer + everything)
- Transport cost in budget = EXACT "الإجمالي" from the recommended transport option in transport table
- Transport table total and budget transport must be the SAME number. If not, FIX IT.
- Total cost in budget < user's budget (with remaining amount shown)
- No embassy info for domestic travel
- No "نظرة عامة" section (info is in app header)
⚠️ THE USER SEES TWO TABS WITH DIFFERENT DATA SOURCES.
Tab 1 shows YOUR output. Tab 4 shows the Budget agent's notes.
Your numbers MUST match the Budget agent's notes EXACTLY.
If you write different numbers, the user sees conflicting data and loses trust.

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
- PRICE REALITY CHECK (2025-2026 prices):
  • Domestic flights in Egypt: minimum 5,000 EGP round trip per person. NEVER under 5,000 EGP.
    Cairo→Sharm: 6,000-13,000 | Cairo→Hurghada: 6,000-12,000 | Cairo→Luxor: 5,000-10,000
  • If transport cost seems unrealistically low (e.g., 2,000 EGP for a flight), FLAG IT and
    replace with realistic price range (e.g., Cairo→Sharm round trip = 6,000-13,000 EGP).
  • International flights from Egypt: Gulf/Turkey 10,000-25,000 EGP, Europe 15,000-35,000 EGP.
  • Private car transfers: 4,000-8,000 EGP for 5-6 hour routes.
  • Bus tickets: 400-800 EGP per person one way.
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
- If REMAINING < 0: FLAG as OVER-BUDGET. This is CRITICAL — the plan CANNOT go forward like this.
  RESOLUTION ORDER:
  1. If transport > 40% of budget → switch to cheaper transport option (economy instead of business, bus instead of flight).
  2. If hotel > 50% of budget → switch to cheaper hotel option.
  3. Cut activities.
  4. If STILL over budget after all cuts → keep the cheapest viable plan and add a note:
     "⚠️ الميزانية لا تكفي لتغطية الرحلة. يُنصح بزيادة الميزانية إلى [X] أو تقليل عدد الأيام."
- TRANSPORT BUDGET CHECK: If TOTAL_TRANSPORT > 40% of user_budget → FLAG IT.
  Switch to cheaper transport even for luxury travelers. A luxury traveler with a small budget
  should get good economy class, not business class that leaves no money for the hotel.
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


def _extract_numbers_from_text(text: str) -> List[float]:
    """Extract all numbers from text, handling Arabic/English formats like 50,000 or 50000."""
    numbers = []
    for match in re.findall(r'[\d,]+(?:\.\d+)?', text):
        try:
            numbers.append(float(match.replace(',', '')))
        except ValueError:
            pass
    return numbers


# Map a currency CODE (as chosen in the form) to its display symbol.
CURRENCY_SYMBOLS = {
    'EGP': 'ج.م', 'USD': '$', 'EUR': '€', 'GBP': '£',
    'SAR': 'ر.س', 'AED': 'د.إ', 'JPY': '¥', 'TRY': '₺',
    'KWD': 'د.ك', 'QAR': 'ر.ق', 'OMR': 'ر.ع', 'BHD': 'د.ب',
}


def _currency_from_question(question: str):
    """Read the user's chosen currency from the request text.
    The frontend always writes a 'Preferred currency: XXX' line. Returns (code, symbol).
    Falls back to USD/$ when nothing is found — NOT to EGP, which was the old mislabel bug."""
    code = None
    if question:
        m = re.search(r'Preferred\s*currency\s*[:\-]?\s*([A-Za-z]{3})', question, re.IGNORECASE)
        if not m:
            m = re.search(r'(?:Total\s*budget|budget)\s*[:\-]?\s*[\d,]+\s*([A-Za-z]{3})', question, re.IGNORECASE)
        if m:
            code = m.group(1).upper()
    if code not in CURRENCY_SYMBOLS:
        # also accept a literal symbol appearing in the question
        for c, sym in CURRENCY_SYMBOLS.items():
            if sym in (question or '') and sym not in ('$',):  # '$' is too ambiguous to trust alone
                code = c
                break
    if code not in CURRENCY_SYMBOLS:
        code = 'USD'
    return code, CURRENCY_SYMBOLS[code]


# Every currency token we might need to rewrite away from. Longer tokens first so
# multi-char tokens (e.g. "ج.م") match before single chars.
_ALL_CUR_TOKENS = ['ج.م', 'ر.س', 'د.إ', 'د.ك', 'ر.ق', 'ر.ع', 'د.ب',
                   'EGP', 'USD', 'EUR', 'GBP', 'SAR', 'AED', 'JPY', 'TRY',
                   'KWD', 'QAR', 'OMR', 'BHD', '€', '£', '₺', '¥', '$']


def _normalize_currency_in_text(text: str, target_sym: str) -> str:
    """Rewrite every money amount in `text` into the user's chosen currency symbol.

    SAFE because all numbers in the plan are computed against the budget, which is ALWAYS in
    the user's chosen currency — so a wrong symbol is a MISLABEL on a correct-magnitude number,
    not a value in another currency. We fix the symbol, we do NOT convert the number.
    (e.g. a USD trip that printed "540 ج.م" for a hotel really means "$540".)
    """
    if not text or not target_sym:
        return text
    others = [t for t in _ALL_CUR_TOKENS if t != target_sym]
    # also don't rewrite the code that maps to the same symbol (e.g. target '$' keeps 'USD')
    same_code = {sym: code for code, sym in CURRENCY_SYMBOLS.items()}.get(target_sym)
    if same_code:
        others = [t for t in others if t != same_code]
    alt = '|'.join(re.escape(t) for t in sorted(others, key=len, reverse=True))
    if not alt:
        return text

    def fmt(num):
        return f"{target_sym}{num}" if target_sym in ('$', '€', '£') else f"{num} {target_sym}"

    # amount THEN token:  "540 ج.م" / "540ج.م" / "2,100 EGP"
    text = re.sub(rf'([\d,]+(?:\.\d+)?)\s*(?:{alt})', lambda m: fmt(m.group(1)), text)
    # token THEN amount:  "$540" / "ج.م 540"
    text = re.sub(rf'(?:{alt})\s*([\d,]+(?:\.\d+)?)', lambda m: fmt(m.group(1)), text)
    # bare leftover tokens with no adjacent number → just the target symbol
    text = re.sub(rf'(?:{alt})', target_sym, text)
    return text


def _extract_total_from_draft(draft: str, user_budget: float) -> dict:
    """
    CODE-LEVEL budget extraction from the draft text.
    Looks for total cost patterns in Arabic and English.
    Returns: {"estimated_total": float or None, "is_over_budget": bool, "overage": float}
    """
    if not draft or not user_budget:
        return {"estimated_total": None, "is_over_budget": False, "overage": 0}

    total = None

    # ── Pattern 1: Direct total patterns (Arabic + English) ──
    # Handles: "الإجمالي الكلي: 108,000", "Total Cost: $6,474", "TOTAL | $6,474"
    total_patterns = [
        # Arabic patterns — number after label
        r'الإجمالي\s*الكلي[:\s=]+\s*\$?([\d,]+)',
        r'إجمالي\s*التكاليف[:\s=]+\s*\$?([\d,]+)',
        r'المجموع\s*الكلي[:\s=]+\s*\$?([\d,]+)',
        # Arabic with currency suffix
        r'الإجمالي[:\s=]+\s*([\d,]+(?:\.\d+)?)\s*(?:ج\.م|EGP|AED|USD|SAR|EUR|GBP|د\.إ|ر\.س)',
        # English patterns — "Total Cost: $6,474" or "Total: $6,474"
        r'Total\s*Cost[:\s=]+\s*\$?([\d,]+)',
        r'(?<!\w)TOTAL[:\s=]+\s*\$?([\d,]+)',
        # $ prefix patterns — "$6,474" after total/TOTAL label
        r'Total[:\s=]+\s*\$([\d,]+(?:\.\d+)?)',
    ]

    for pattern in total_patterns:
        match = re.search(pattern, draft, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(',', ''))
                if val > 500:  # lowered from 1000 for USD amounts
                    total = val
                    break
            except ValueError:
                pass

    # ── Pattern 2: Look in budget table (HTML or markdown pipe table) ──
    if total is None:
        table_patterns = [
            # HTML table: <td> after TOTAL/الإجمالي row
            r'الإجمالي\s*الكلي.*?<td[^>]*>\s*\$?([\d,]+)',
            r'(?:\*\*)?TOTAL(?:\*\*)?.*?<td[^>]*>\s*(?:\*\*)?\$?([\d,]+)',
            r'Total.*?<td[^>]*>\s*\$?([\d,]+)',
            # Markdown pipe table: | TOTAL | $6,474 | or | **TOTAL** | **$6,474** |
            # "الكلي" is optional — the app often renders just "الإجمالي".
            r'\|\s*(?:\*\*)?(?:TOTAL|الإجمالي(?:\s*الكلي)?|إجمالي)(?:\*\*)?\s*\|\s*(?:\*\*)?\$?\s*([\d,]+)',
            # Plain line (no pipes): "الإجمالي $9,144" / "الإجمالي: $9,144"
            r'الإجمالي(?:\s*الكلي)?\s*[:\s=]*\$\s*([\d,]+)',
        ]
        for pattern in table_patterns:
            match = re.search(pattern, draft, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    val = float(match.group(1).replace(',', ''))
                    if val > 500:
                        total = val
                        break
                except ValueError:
                    pass

    # ── Pattern 3: Sum approach — look for transport + hotel + activities ──
    if total is None:
        transport_total = None
        hotel_total = None
        activities_total = None

        for pattern in [
            r'إجمالي\s*(?:تكلفة\s*)?النقل[:\s=]+\s*\$?([\d,]+)',
            r'[Tt]ransport[:\s=]+\s*\$?([\d,]+)',
            r'\|\s*Transport\b.*?\|\s*\$?([\d,]+)',
        ]:
            m = re.search(pattern, draft, re.IGNORECASE)
            if m:
                try: transport_total = float(m.group(1).replace(',', ''))
                except: pass
                break

        for pattern in [
            r'إجمالي\s*(?:تكلفة\s*)?الفناد[قك][:\s=]+\s*\$?([\d,]+)',
            r'[Hh]otel[:\s]*(?:total)?[:\s=]+\s*\$?([\d,]+)',
            r'\|\s*Hotel\b.*?\|\s*\$?([\d,]+)',
        ]:
            m = re.search(pattern, draft, re.IGNORECASE)
            if m:
                try: hotel_total = float(m.group(1).replace(',', ''))
                except: pass
                break

        for pattern in [
            r'[Aa]ctivities[:\s=]+\s*\$?([\d,]+)',
            r'\|\s*Activities\b.*?\|\s*\$?([\d,]+)',
        ]:
            m = re.search(pattern, draft, re.IGNORECASE)
            if m:
                try: activities_total = float(m.group(1).replace(',', ''))
                except: pass
                break

        parts = [x for x in [transport_total, hotel_total, activities_total] if x]
        if len(parts) >= 2:
            total = sum(parts)

    # ── Pattern 4: NEGATIVE "الميزانية المتبقية" (remaining budget) is the most robust
    #    over-budget signal — the app prints it as -$2,144 when the plan overshoots.
    #    From it we can recover the total: total = budget + |remaining|.
    if total is None or total <= user_budget:
        # Grab whatever follows the "remaining budget" label up to the row/line end
        # (handles pipe tables "| الميزانية المتبقية | -$2,144 |" and plain lines alike).
        rem_m = re.search(
            r'(?:الميزانية\s*المتبقية|المتبقّ?ي|Remaining(?:\s*Budget)?)\s*[:\s=|]*([^\n|]{0,24})',
            draft, re.IGNORECASE)
        if rem_m:
            token = rem_m.group(1)
            # Negative remaining (minus sign or parentheses) ⇒ over budget.
            if re.search(r'[-−(]', token):
                num = re.sub(r'[^\d]', '', token)
                try:
                    deficit = float(num) if num else 0
                    if deficit > 0:
                        recovered_total = user_budget + deficit
                        if total is None or recovered_total > total:
                            total = recovered_total
                except ValueError:
                    pass

    is_over = total is not None and total > user_budget
    overage = (total - user_budget) if is_over and total else 0

    return {"estimated_total": total, "is_over_budget": is_over, "overage": overage}


def _filter_over_budget_options(notes: List[str], max_price: float, category: str) -> List[str]:
    """
    CODE-LEVEL: Remove or annotate options from research notes that exceed the budget limit.
    This ensures the LLM never sees over-budget options, so it CAN'T pick them.

    Args:
        notes: List of research note strings (from flight_notes, hotel_notes, etc.)
        max_price: Maximum allowed price per unit (per person for flights, per night for hotels)
        category: "transport" or "hotel" — determines what patterns to look for
    Returns:
        Filtered notes with over-budget options removed or replaced with warnings
    """
    if not notes or not max_price or max_price <= 0:
        return notes

    filtered = []
    for note in notes:
        # Extract all prices from the note
        prices = re.findall(r'(?:\$|USD\s*|EGP\s*|ج\.م\s*|AED\s*|د\.إ\s*|SAR\s*|ر\.س\s*|EUR\s*|€\s*|£\s*)([\d,]+(?:\.\d+)?)|(\d[\d,]+(?:\.\d+)?)\s*(?:\$|USD|EGP|ج\.م|AED|د\.إ|SAR|ر\.س|EUR|€|£)', note, re.IGNORECASE)

        has_over_budget = False
        for price_groups in prices:
            price_str = price_groups[0] or price_groups[1]
            if price_str:
                try:
                    price_val = float(price_str.replace(',', ''))
                    # Only flag prices that are clearly per-unit (not too small)
                    if price_val > 100 and price_val > max_price * 1.1:
                        has_over_budget = True
                        break
                except ValueError:
                    pass

        if has_over_budget:
            # Check if this is business/first class (for transport) or luxury hotel
            is_premium = any(kw in note.lower() for kw in [
                'business', 'first class', 'بيزنس', 'درجة أولى', 'first',
                'suite', 'سويت', 'presidential', 'رئاسي', 'villa', 'فيلا'
            ])
            if is_premium:
                # Replace with warning — don't include the option at all
                filtered.append(
                    f"⛔ [REMOVED - OVER BUDGET] An option was removed because it exceeds "
                    f"the max {category} budget of {int(max_price):,}. Only options within budget are shown."
                )
            else:
                # Non-premium but expensive — keep but annotate
                filtered.append(f"⚠️ [WARNING: EXCEEDS BUDGET LIMIT OF {int(max_price):,}] {note}")
        else:
            filtered.append(note)

    return filtered


def _generate_budget_notes_from_draft(draft: str, user_budget: float, currency: str) -> List[str]:
    """
    Generate budget_notes directly FROM the draft text, so tab4 always matches tab1.
    This replaces the Budget agent's separate output for the budget tab.
    """
    if not draft:
        return []

    notes = []

    # Extract key cost sections from the draft
    transport_match = re.search(r'إجمالي\s*(?:تكلفة\s*)?النقل[:\s=]+\s*([\d,]+)', draft)
    hotel_match = re.search(r'إجمالي\s*(?:تكلفة\s*)?الفناد[قك].*?[:\s=]+\s*([\d,]+)', draft)
    total_match = re.search(r'الإجمالي\s*الكلي[:\s=]+\s*([\d,]+)', draft)
    remaining_match = re.search(r'الميزانية\s*المتبقية[:\s=]+\s*(-?[\d,]+)', draft)

    if transport_match:
        notes.append(f"تكلفة النقل: {transport_match.group(1)} {currency}")
    if hotel_match:
        notes.append(f"تكلفة الفنادق: {hotel_match.group(1)} {currency}")
    if total_match:
        notes.append(f"الإجمالي الكلي: {total_match.group(1)} {currency}")
    if remaining_match:
        notes.append(f"الميزانية المتبقية: {remaining_match.group(1)} {currency}")

    if user_budget:
        notes.append(f"ميزانية المستخدم: {int(user_budget):,} {currency}")

    return notes


# Keywords that mark a line as a (paid) meal in the itinerary.
_MEAL_KW = ['dinner', 'lunch', 'breakfast', 'brunch', 'meal', 'dine', 'dining',
            'restaurant', 'ristorante', 'trattoria', 'osteria', 'bistro',
            'عشاء', 'غداء', 'فطور', 'إفطار', 'وجبة', 'وجبات', 'مطعم', 'مطاعم',
            'تناول', 'عشا', 'غدا', 'مأكولات']
_PP_KW = ['/person', 'per person', 'per-person', ' pp', 'للفرد', '/فرد', 'للشخص', 'لكل شخص', 'للفرد الواحد']
_GROUP_KW = ['for the group', 'للمجموعة', 'للمجموعه', 'للجميع', 'group total', 'total for the']
_PRICE_RE = re.compile(
    r'(?:\$|USD|€|£|ج\.م|EGP|AED|د\.إ|SAR|ر\.س)\s*([\d,]+(?:\.\d+)?)'
    r'|([\d,]+(?:\.\d+)?)\s*(?:\$|USD|€|£|ج\.م|EGP|AED|د\.إ|SAR|ر\.س)', re.IGNORECASE)


def _sum_itinerary_food(text: str, num_travelers: int) -> float:
    """Sum the cost of PAID meals scheduled in the day-by-day itinerary.
    Per-person meals are multiplied by travelers; group/unspecified meals counted once.
    Used to detect a 'Food: 0 but Michelin dinners listed' contradiction."""
    if not text:
        return 0.0
    nt = max(int(num_travelers or 1), 1)
    total = 0.0
    for raw in text.splitlines():
        low = raw.lower()
        if not any(k in low for k in _MEAL_KW):
            continue
        # skip the budget-table row itself (we want itinerary meals, not the summary line)
        if ('الطعام' in raw or 'food' in low) and ('|' in raw or 'budget' in low):
            continue
        m = _PRICE_RE.search(raw)
        if not m:
            continue
        val_s = m.group(1) or m.group(2)
        try:
            val = float(val_s.replace(',', ''))
        except (ValueError, AttributeError):
            continue
        if val <= 0:
            continue
        is_pp = any(k in low for k in _PP_KW)
        is_group = any(k in low for k in _GROUP_KW)
        total += val * nt if (is_pp and not is_group) else val
    return total


def _extract_food_line(text: str):
    """Return the Food/الطعام amount from the budget table, or None if there's no food row.
    '0 (مشمول في الفندق)' / 'included' returns 0.0."""
    if not text:
        return None
    for pat in [
        r'\|\s*(?:\*\*)?(?:الطعام|الأكل|الوجبات|Food|Dining|Meals)(?:\*\*)?[^\|\n]*\|\s*(?:\*\*)?\$?\s*([\d,]+)',
        r'(?:الطعام|الأكل|الوجبات|Food|Dining|Meals)\s*(?:\([^)]*\))?\s*[:=]\s*\$?\s*([\d,]+)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                pass
    # A food row that says 0 / included / مشمول
    if re.search(r'(?:الطعام|Food)\b[^\n|]*(?:\|\s*)?(?:\$?\s*0\b|included|مشمول|مجان)', text, re.IGNORECASE):
        return 0.0
    return None


def _inject_budget_compliant_option(notes: List[str], max_price: float, category: str, destination: str) -> List[str]:
    """
    CODE-LEVEL: After research agent returns, check if ANY option is within budget.
    If ALL options are over budget, inject a realistic budget-compliant option.
    This guarantees the LLM always has at least one option within budget to pick.
    """
    if not notes or not max_price or max_price <= 0:
        return notes

    has_within_budget = False
    for note in notes:
        prices = re.findall(r'(?:\$|USD\s*)([\d,]+(?:\.\d+)?)|(\d[\d,]+(?:\.\d+)?)\s*(?:\$|USD)', note, re.IGNORECASE)
        for price_groups in prices:
            price_str = price_groups[0] or price_groups[1]
            if price_str:
                try:
                    price_val = float(price_str.replace(',', ''))
                    if 50 < price_val <= max_price * 1.1:
                        has_within_budget = True
                        break
                except ValueError:
                    pass
        if has_within_budget:
            break

    if not has_within_budget:
        if category == "transport":
            notes.append(
                f"✅ [BUDGET-COMPLIANT OPTION] Economy class flight to {destination}: "
                f"estimated ${int(max_price * 0.7):,}-${int(max_price):,} per person round trip. "
                f"Airlines: budget carriers or economy class on major airlines. "
                f"Search for the cheapest available option. "
                f"Maximum allowed: ${int(max_price):,}/person. "
                f"DO NOT exceed this price."
            )
        elif category == "hotel":
            star_rating = "4-star" if max_price > 150 else "3-star"
            notes.append(
                f"✅ [BUDGET-COMPLIANT OPTION] {star_rating} hotel in {destination}: "
                f"estimated ${int(max_price * 0.5):,}-${int(max_price):,} per night. "
                f"Good location, clean, well-rated. "
                f"Maximum allowed: ${int(max_price):,}/night. "
                f"If budget only allows a {star_rating}, recommend that. "
                f"Tell user: لو عايز فندق أفخم، يُنصح بزيادة الميزانية."
            )
        print(f"[CODE-LEVEL] Injected budget-compliant {category} option (max: {int(max_price):,}) — all research options were over budget")

    return notes


def _cheapest_unit_price(notes: List[str], kind: str):
    """Scan research notes and return the cheapest realistic per-unit price found
    (per person for transport, per night for hotel), plus a short source label.
    Returns (price: float|None, label: str|None)."""
    if not notes:
        return None, None
    floor = 100 if kind == "transport" else 30   # ignore ratings/durations/"4 ليالٍ"
    ceil = 50000                                   # ignore absurd parses
    best, best_label = None, None
    price_re = re.compile(
        r'(?:\$|USD\s*|EGP\s*|ج\.م\s*|AED\s*|د\.إ\s*|SAR\s*|ر\.س\s*|EUR\s*|€\s*|£\s*)([\d,]+(?:\.\d+)?)'
        r'|(\d[\d,]+(?:\.\d+)?)\s*(?:\$|USD|EGP|ج\.م|AED|د\.إ|SAR|ر\.س|EUR|€|£)',
        re.IGNORECASE)
    for note in notes:
        up = note.upper()
        if '⛔' in note or 'REMOVED' in up or 'OVER BUDGET' in up:
            continue  # skip the "removed over-budget" placeholder lines
        for g in price_re.findall(note):
            s = g[0] or g[1]
            if not s:
                continue
            try:
                v = float(s.replace(',', ''))
            except ValueError:
                continue
            if floor <= v <= ceil and (best is None or v < best):
                best, best_label = v, note.strip()[:120]
    return best, best_label


def _priciest_unit_price(notes: List[str], kind: str, ceiling=None):
    """Scan research notes and return the most expensive realistic per-unit price found
    at or below `ceiling` (if given), plus a short source label. Used to pick the BEST
    hotel the budget can afford once transport has been minimized."""
    if not notes:
        return None, None
    floor = 100 if kind == "transport" else 30
    ceil = 50000
    best, best_label = None, None
    price_re = re.compile(
        r'(?:\$|USD\s*|EGP\s*|ج\.م\s*|AED\s*|د\.إ\s*|SAR\s*|ر\.س\s*|EUR\s*|€\s*|£\s*)([\d,]+(?:\.\d+)?)'
        r'|(\d[\d,]+(?:\.\d+)?)\s*(?:\$|USD|EGP|ج\.م|AED|د\.إ|SAR|ر\.س|EUR|€|£)',
        re.IGNORECASE)
    for note in notes:
        up = note.upper()
        if '⛔' in note or 'REMOVED' in up or 'OVER BUDGET' in up:
            continue
        for g in price_re.findall(note):
            s = g[0] or g[1]
            if not s:
                continue
            try:
                v = float(s.replace(',', ''))
            except ValueError:
                continue
            if floor <= v <= ceil and (ceiling is None or v <= ceiling) and (best is None or v > best):
                best, best_label = v, note.strip()[:120]
    return best, best_label


def _fmt_money(value, currency: str) -> str:
    """Format an amount with the currency symbol in the right place."""
    v = int(round(value))
    if currency in ('$', '€', '£'):
        return f"{currency}{v:,}"
    return f"{v:,} {currency}"


def _build_budget_section(transport, hotel, activities, num_nights, user_budget,
                          currency, note: str) -> str:
    """Build a clean, authoritative Arabic budget section with correct math."""
    total = transport + hotel + activities
    remaining = user_budget - total
    rem_str = _fmt_money(remaining, currency)
    if remaining < 0:
        rem_str = "-" + _fmt_money(abs(remaining), currency)
    nights = max(int(num_nights or 1), 1)
    night = hotel / nights
    lines = [
        "## 💰 الميزانية الإجمالية",
        "",
        "| البند | التكلفة | التفاصيل |",
        "|---|---|---|",
        f"| النقل (كل المسافرين) | {_fmt_money(transport, currency)} | — |",
        f"| الإقامة ({nights} ليالٍ) | {_fmt_money(hotel, currency)} | {_fmt_money(night, currency)} × {nights} |",
        f"| الأنشطة والتجارب | {_fmt_money(activities, currency)} | — |",
        f"| **الإجمالي الكلي** | **{_fmt_money(total, currency)}** | — |",
        f"| الميزانية المتاحة | {_fmt_money(user_budget, currency)} | — |",
        f"| الميزانية المتبقية | {rem_str} | — |",
        "",
        note,
    ]
    return "\n".join(lines)


def _parse_hotel_table_rows(draft: str):
    """Parse the hotel comparison table into [(name, price_per_night), …] from the draft.
    Reads the actual table rows, skipping header/separator rows and the rating/stars columns.
    This is the STRICT source of truth for hotel prices — never guessed."""
    rows = []
    price_re = re.compile(
        r'([\d][\d,]{2,}(?:\.\d+)?)\s*(?:ج\.م|EGP|AED|د\.إ|SAR|ر\.س|EUR|€|£|USD|\$)'
        r'|(?:\$|€|£)\s*([\d][\d,]{2,}(?:\.\d+)?)')
    for line in draft.splitlines():
        if line.count('|') < 3:                       # need a real table row
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        joined = ' '.join(cells).lower()
        if ('الفندق' in joined and ('السعر' in joined or 'النجوم' in joined)) or 'hotel name' in joined:
            continue                                  # header row
        if set(''.join(cells)) <= set('-: *'):        # markdown separator row
            continue
        price = None
        for c in cells:
            pm = price_re.search(c)
            if pm:
                try:
                    price = float((pm.group(1) or pm.group(2)).replace(',', ''))
                    break
                except ValueError:
                    pass
        if price and 300 <= price <= 100000 and cells and cells[0]:
            rows.append((cells[0], price))
    return rows


def _recommended_hotel_night_price(draft: str, hotel_notes, travel_style: str):
    """STRICT per-night price of the recommended hotel. Reads the hotel TABLE and matches the
    🏆-recommended hotel to its row, so the budget's hotel line can ALWAYS equal
    (recommended hotel price/night) × nights. Only returns None when no hotel table exists at all.
    This is what stops the recurring '75,000 instead of 85,000' bug."""
    try:
        rows = _parse_hotel_table_rows(draft)
        if rows:
            # 1) match the 🏆-recommended hotel name to a table row
            mrec = re.search(r'🏆[^\n:]*:\s*([^\n—\-–|]{3,80})', draft)
            if mrec:
                name = mrec.group(1).strip()
                # exact / substring match first
                for rname, price in rows:
                    if rname and (rname in name or name in rname):
                        return price
                # token-overlap match (handles minor wording differences)
                nt = set(re.findall(r'[\w؀-ۿ]+', name))
                best, best_ov = None, 0
                for rname, price in rows:
                    ov = len(nt & set(re.findall(r'[\w؀-ۿ]+', rname)))
                    if ov > best_ov:
                        best_ov, best = ov, price
                if best is not None and best_ov >= 1:
                    return best
            # 2) no usable 🏆 match → choose by style directly from the real table prices
            prices = sorted(p for _, p in rows)
            style = _style_key(travel_style)
            if style == 'luxury':
                return prices[-1]          # best hotel the plan listed
            if style == 'budget':
                return prices[0]
            return prices[len(prices) // 2]
        # 3) no table found → last-resort fallbacks (old behaviour)
        price_re = re.compile(
            r'([\d][\d,]{2,})\s*(?:ج\.م|EGP|AED|د\.إ|SAR|ر\.س|EUR|€|£|USD)'
            r'|(?:\$|€|£)\s*([\d][\d,]{2,})')
        mrec = re.search(r'🏆[^\n:]*:\s*([^\n—\-–|]{3,60})', draft)
        if mrec:
            name = mrec.group(1).strip()
            idx = draft.find(name)
            if idx != -1:
                pm = price_re.search(draft[idx: idx + 220])
                if pm:
                    val = float((pm.group(1) or pm.group(2)).replace(',', ''))
                    if 300 <= val <= 50000:
                        return val
        if _style_key(travel_style) == 'luxury':
            p, _ = _priciest_unit_price(hotel_notes, "hotel")
            return p
    except Exception:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC HOTEL PRICE SANITY BANDS — realistic per-night price ranges
# by destination & star tier. The LLM's hotel prices are CLAMPED into these
# bands in code, so a hallucinated 2,079 EGP for a 5-star in Dahab (real
# ~6,400) is pulled up to a realistic figure, and an inflated 4,000 for a
# cheap hotel is pulled down. Not the exact live price (that needs a hotel
# API) but NEVER wildly wrong. EGP bands for Egyptian cities; international
# destinations use USD bands × FX. (low, high) per night.
# ═══════════════════════════════════════════════════════════════════════

_HOTEL_BAND_EGP = {  # Egyptian city group -> star -> (low, high) EGP/night
    'dahab':         {3: (1500, 3200), 4: (2800, 5500), 5: (4800, 9500)},
    'sharm el sheikh': {3: (2000, 4500), 4: (3500, 9000), 5: (5500, 15000)},
    'hurghada':      {3: (1800, 3800), 4: (3000, 6800), 5: (4500, 12000)},
    'el gouna':      {3: (2500, 5000), 4: (4000, 9000), 5: (7000, 16000)},
    'sahl hasheesh': {3: (3000, 6000), 4: (5000, 10000), 5: (8000, 20000)},
    'soma bay':      {3: (3000, 6000), 4: (5000, 10000), 5: (8000, 19000)},
    'makadi':        {3: (2500, 5000), 4: (4000, 8500), 5: (6500, 15000)},
    'marsa alam':    {3: (2000, 4200), 4: (3500, 8000), 5: (5000, 14000)},
    'cairo':         {3: (1500, 3500), 4: (3000, 6800), 5: (5000, 13000)},
    'alexandria':    {3: (1500, 3200), 4: (2800, 6000), 5: (4500, 11000)},
    'luxor':         {3: (1500, 3300), 4: (3000, 6500), 5: (4500, 12000)},
    'aswan':         {3: (1500, 3300), 4: (3000, 6500), 5: (4500, 12000)},
    'taba':          {3: (2000, 4000), 4: (3500, 7000), 5: (5000, 12000)},
    'nuweiba':       {3: (1500, 3200), 4: (2800, 5500), 5: (4500, 9000)},
    '_default':      {3: (1500, 3500), 4: (3000, 7000), 5: (5000, 13000)},
}

_HOTEL_BAND_USD = {  # international region -> star -> (low, high) USD/night
    'europe':        {3: (90, 180), 4: (150, 340), 5: (280, 850)},
    'gulf':          {3: (70, 150), 4: (130, 300), 5: (220, 700)},
    'turkey':        {3: (60, 130), 4: (110, 260), 5: (180, 550)},
    'levant':        {3: (60, 130), 4: (110, 240), 5: (180, 500)},
    'north_africa':  {3: (60, 130), 4: (110, 250), 5: (180, 520)},
    'east_africa':   {3: (70, 150), 4: (130, 300), 5: (220, 650)},
    'sea':           {3: (35, 90),  4: (90, 230),  5: (160, 600)},
    'south_asia':    {3: (40, 100), 4: (90, 240),  5: (170, 650)},
    'east_asia':     {3: (70, 160), 4: (130, 320), 5: (230, 750)},
    'north_america': {3: (90, 180), 4: (150, 350), 5: (260, 800)},
    'oceania':       {3: (90, 190), 4: (150, 360), 5: (270, 820)},
    '_default':      {3: (70, 160), 4: (120, 300), 5: (220, 700)},
}


def _hotel_band(destination: str, stars: int, currency: str):
    """Return (low, high) realistic per-night price for destination+star tier, in the
    trip currency. None when we can't place the destination (then no clamp happens)."""
    star = 5 if stars >= 5 else (3 if stars <= 3 else 4)
    is_egp = ('ج.م' in (currency or '') or 'EGP' in (currency or '').upper())
    dest = (destination or '')
    if is_egp:
        city = _norm_city(dest)
        table = _HOTEL_BAND_EGP.get(city)
        if not table:
            # only clamp with the generic Egypt band if it really looks Egyptian
            if _intl_region(dest) is not None:
                return None  # it's international but priced in EGP — skip (rare)
            table = _HOTEL_BAND_EGP['_default']
        return table.get(star)
    # non-EGP → international region band × FX
    region = _intl_region(dest)
    fx = _USD_FX.get((currency or '$').strip())
    if not fx:
        return None
    table = _HOTEL_BAND_USD.get(region or '_default')
    lo, hi = table.get(star)
    return (lo * fx, hi * fx)


def _clamp_hotel_prices_in_draft(draft: str, destination: str, currency: str):
    """Clamp every hotel-table nightly price into its realistic band for the destination &
    star tier. Returns (new_draft, changed). Deterministic — stops hotel-price hallucination."""
    if not draft:
        return draft, False
    price_re = re.compile(
        r'([\d][\d,]{2,}(?:\.\d+)?)\s*(?:ج\.م|EGP|AED|د\.إ|SAR|ر\.س|EUR|€|£|USD|\$)'
        r'|(?:\$|€|£)\s*([\d][\d,]{2,}(?:\.\d+)?)')
    changed = False
    out = []
    for line in draft.splitlines():
        if line.count('|') < 3:
            out.append(line); continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        joined = ' '.join(cells).lower()
        if ('الفندق' in joined and ('السعر' in joined or 'النجوم' in joined)) or 'hotel name' in joined \
           or set(''.join(cells)) <= set('-: *'):
            out.append(line); continue
        pm = price_re.search(line)
        if not pm:
            out.append(line); continue
        try:
            price = float((pm.group(1) or pm.group(2)).replace(',', ''))
        except ValueError:
            out.append(line); continue
        if not (200 <= price <= 300000):
            out.append(line); continue
        stars = line.count('⭐') + line.count('★')
        if stars == 0:
            stars = 4  # assume mid-tier when stars aren't shown
        band = _hotel_band(destination, stars, currency)
        if not band:
            out.append(line); continue
        lo, hi = band
        new_price = min(max(price, lo), hi)
        if abs(new_price - price) > max(1, price * 0.02):
            old_num = pm.group(1) or pm.group(2)
            line = line.replace(old_num, f"{int(round(new_price)):,}", 1)
            changed = True
        out.append(line)
    return "\n".join(out), changed


# ═══════════════════════════════════════════════════════════════════════
# REAL RECOMMENDED-HOTEL PRICE FROM LIVE SEARCH — the bands above only keep a
# price "realistic"; they can't produce the TRUE price. So for the ONE hotel the
# plan recommends (the hotel that drives the budget and the one the user checks
# against Google), we pull the actual nightly price from live Tavily search and
# use it — clamped to the band only as a safety net. Free (uses the Tavily key
# already in the app). Best-effort: when search returns no usable price, we fall
# back to the band-clamped value, so this is NEVER worse than before.
# ═══════════════════════════════════════════════════════════════════════

_CCY_WORD_TO_SYM = {
    'usd': '$', 'us$': '$', 'dollar': '$', 'dollars': '$',
    'egp': 'ج.م', 'le': 'ج.م', 'جنيه': 'ج.م', 'ج.م': 'ج.م',
    'egyptian pound': 'ج.م', 'egyptian pounds': 'ج.م', 'pound': 'ج.م', 'pounds': 'ج.م',
    'eur': '€', 'euro': '€', 'euros': '€',
    'gbp': '£', 'sar': 'ر.س', 'riyal': 'ر.س', 'riyals': 'ر.س', 'ر.س': 'ر.س',
    'aed': 'د.إ', 'dirham': 'د.إ', 'dirhams': 'د.إ', 'د.إ': 'د.إ',
}


def _tavily_text(res) -> str:
    """Flatten a TavilySearchResults result (list of dicts, or a string) into one text blob."""
    if isinstance(res, str):
        return res
    if isinstance(res, list):
        parts = []
        for r in res:
            if isinstance(r, dict):
                parts.append(f"{r.get('content', '')} {r.get('title', '')}")
            else:
                parts.append(str(r))
        return "  ".join(parts)
    return str(res or "")


def _parse_night_prices_usd(text: str, lo_usd: float = 20, hi_usd: float = 3000):
    """Pull price-looking numbers out of search text and normalise each to USD, using the
    currency symbol/word next to the number. Returns a list of USD floats within [lo_usd, hi_usd]
    (defaults suit a nightly hotel rate; widen the bounds for round-trip flights)."""
    if not text:
        return []
    low = text.lower()
    out = []

    def _add(amount_str, sym):
        try:
            amt = float(amount_str.replace(',', ''))
        except ValueError:
            return
        fx = _USD_FX.get(sym)
        if not fx:
            return
        usd = amt / fx
        if lo_usd <= usd <= hi_usd:
            out.append(usd)

    # symbol BEFORE the number:  $120 / US$120 / €95 / £80
    for m in re.finditer(r'(us\$|\$|€|£)\s*([\d][\d,]{1,6}(?:\.\d{1,2})?)', low):
        _add(m.group(2), {'us$': '$', '$': '$', '€': '€', '£': '£'}[m.group(1)])
    # number BEFORE a currency word/code:  6,500 EGP / 120 usd / 95 euro / 450 egyptian pounds
    for m in re.finditer(
        r'([\d][\d,]{1,6}(?:\.\d{1,2})?)\s*'
        r'(egyptian pounds?|us\$|usd|egp|eur|euros?|gbp|sar|aed|dollars?|pounds?|riyals?|dirhams?|le|جنيه|ج\.م|ر\.س|د\.إ)',
        low):
        tok = m.group(2).strip()
        sym = _CCY_WORD_TO_SYM.get(tok)
        if sym:
            _add(m.group(1), sym)
    return out


_HOTEL_PRICE_CACHE = {}   # (name, dest, currency) -> price or None — one live search per hotel per run


def _real_hotel_night_price(hotel_name: str, destination: str, currency: str):
    """Live per-night price for a specific hotel, in the trip currency, via Tavily search.
    Runs several query variants, pools every price found, trims promo 'from' low-outliers, and
    returns the MEDIAN (the typical real rate). Returns None only when search finds no price at
    all — so the caller can show a realistic estimate instead of a hallucinated lowball.
    Cached per (hotel, destination, currency) so the review loop and finalizer share one search."""
    try:
        if not hotel_name:
            return None
        fx = _USD_FX.get((currency or '$').strip())
        if not fx:
            return None
        dest = (destination or '').split(',')[0].strip()
        ckey = (hotel_name.strip().lower(), dest.lower(), (currency or '$').strip())
        if ckey in _HOTEL_PRICE_CACHE:
            return _HOTEL_PRICE_CACHE[ckey]
        queries = [
            f"{hotel_name} {dest} price per night",
            f"{hotel_name} {dest} booking price per night 2026",
            f"{hotel_name} سعر الليلة",
        ]
        pooled = []
        for q in queries:
            try:
                pooled += _parse_night_prices_usd(_tavily_text(search_tool.invoke(q.strip())))
            except Exception:
                continue
            if len(pooled) >= 6:         # enough signal — stop searching early
                break
        if not pooled:
            _HOTEL_PRICE_CACHE[ckey] = None
            return None
        pooled.sort()
        median_usd = pooled[len(pooled) // 2]
        # drop promotional 'from' prices far below the median (keep 55%..300% of median)
        core = [p for p in pooled if 0.55 * median_usd <= p <= 3.0 * median_usd]
        if not core:
            core = pooled
        core.sort()
        typical_usd = core[len(core) // 2]
        result = typical_usd * fx
        _HOTEL_PRICE_CACHE[ckey] = result
        return result
    except Exception:
        return None


def _apply_real_recommended_price(draft: str, destination: str, currency: str):
    """Rewrite the 🏆-recommended hotel's table price to the REAL live price (clamped to its
    band). Downstream budget math then uses this real figure. Returns (new_draft, changed)."""
    try:
        if not draft:
            return draft, False
        mrec = re.search(r'🏆[^\n:]*:\s*([^\n—\-–|]{3,80})', draft)
        if not mrec:
            return draft, False
        rec_name = mrec.group(1).strip()
        rec_tokens = set(re.findall(r'[\w؀-ۿ]{2,}', rec_name))
        price_re = re.compile(
            r'([\d][\d,]{2,}(?:\.\d+)?)\s*(?:ج\.م|EGP|AED|د\.إ|SAR|ر\.س|EUR|€|£|USD|\$)'
            r'|(?:\$|€|£)\s*([\d][\d,]{2,}(?:\.\d+)?)')
        lines = draft.splitlines()
        for i, line in enumerate(lines):
            if line.count('|') < 3:
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            joined = ' '.join(cells).lower()
            if ('الفندق' in joined and ('السعر' in joined or 'النجوم' in joined)) or 'hotel name' in joined:
                continue
            if set(''.join(cells)) <= set('-: *'):
                continue
            name_cell = cells[0] if cells else ''
            name_tokens = set(re.findall(r'[\w؀-ۿ]{2,}', name_cell))
            if not (name_cell and (name_cell in rec_name or rec_name in name_cell
                                   or len(rec_tokens & name_tokens) >= 1)):
                continue
            pm = price_re.search(line)
            if not pm:
                continue
            try:
                old = float((pm.group(1) or pm.group(2)).replace(',', ''))
            except ValueError:
                continue
            if not (200 <= old <= 300000):
                continue
            real = _real_hotel_night_price(name_cell, destination, currency)
            stars = line.count('⭐') + line.count('★') or 4
            band = _hotel_band(destination, stars, currency)
            if real:
                # Live price found → trust it; the band only guards against freak extremes.
                new_price = min(max(real, band[0]), band[1]) if band else real
                badge = (f"> 🔎 تم التحقق من سعر الفندق مقابل بحث مباشر "
                         f"(~{int(round(new_price)):,} {currency}/ليلة) — price verified against live search.")
            elif band:
                # Search miss → DON'T keep the LLM's possible lowball. Lift to a realistic
                # typical price (band midpoint) for this star tier & city, so the headline
                # hotel is never shown far below reality (the '4,800 for an 8,000 resort' bug).
                lo, hi = band
                realistic = lo + 0.5 * (hi - lo)
                new_price = min(max(old, realistic), hi)
                badge = ("> 🔎 تقدير واقعي للسعر (البحث المباشر لم يرجّع سعرًا مؤكدًا لهذا الفندق) "
                         "— realistic market estimate.")
            else:
                return draft, False
            # rewrite the shown price only when it is materially different from the real/estimate
            if abs(new_price - old) > max(1, old * 0.05):
                old_num = pm.group(1) or pm.group(2)
                lines[i] = line.replace(old_num, f"{int(round(new_price)):,}", 1)
            # surface the verification UNDER the hotel recommendation line, so the user can SEE
            # that the reviewer checked the price against a live search (idempotent — refreshes
            # an existing badge instead of stacking).
            for bi, bl in enumerate(lines):
                if '🏆' in bl and (name_cell in bl or rec_name in bl
                                   or len(rec_tokens & set(re.findall(r'[\w؀-ۿ]{2,}', bl))) >= 1):
                    if bi + 1 < len(lines) and '🔎' in lines[bi + 1]:
                        lines[bi + 1] = badge
                    else:
                        lines.insert(bi + 1, badge)
                    break
            return "\n".join(lines), True
        return draft, False
    except Exception:
        return draft, False


def _verify_recommended_hotel_price(draft: str, destination: str, currency: str,
                                    travel_style: str = "mid-range"):
    """REVIEW-LOOP CHECK: live-search the recommended hotel and compare its real price to the
    price shown in the draft. Returns a dict {name, displayed, real, mismatch, note}. mismatch is
    True when the shown price differs from reality by more than ~22% — the review loop then sends
    the plan back to be corrected with the exact real figure."""
    out = {"name": None, "displayed": None, "real": None, "mismatch": False, "note": ""}
    try:
        if not draft:
            return out
        rows = _parse_hotel_table_rows(draft)
        mrec = re.search(r'🏆[^\n:]*:\s*([^\n—\-–|]{3,80})', draft)
        name = displayed = None
        if mrec and rows:
            rn = mrec.group(1).strip()
            rt = set(re.findall(r'[\w؀-ۿ]{2,}', rn))
            for rname, price in rows:
                if rname and (rname in rn or rn in rname
                              or len(rt & set(re.findall(r'[\w؀-ۿ]{2,}', rname))) >= 1):
                    name, displayed = rname, price
                    break
        if displayed is None:
            displayed = _recommended_hotel_night_price(draft, None, travel_style)
        if name is None and mrec:
            name = mrec.group(1).strip()
        out["name"], out["displayed"] = name, displayed
        if not name or not displayed:
            out["note"] = "تعذّر تحديد الفندق المرشّح للتحقق من سعره."
            return out
        real = _real_hotel_night_price(name, destination, currency)
        out["real"] = real
        if not real:
            out["note"] = f"لم يرجع البحث سعرًا مؤكدًا لـ {name} — تم الإبقاء على التقدير الواقعي."
            return out
        hi, lo = max(real, displayed), min(real, displayed)
        if lo > 0 and (hi / lo) > 1.22:
            out["mismatch"] = True
            out["note"] = (f"سعر {name} المعروض {int(displayed):,} {currency}/ليلة يختلف عن السعر الحقيقي "
                           f"من البحث (~{int(real):,} {currency}/ليلة) — إعادة الضبط على الرقم الحقيقي.")
        else:
            out["note"] = (f"تم التحقق من سعر {name} مقابل البحث المباشر: "
                           f"{int(displayed):,} ≈ {int(real):,} {currency}/ليلة ✅")
    except Exception:
        pass
    return out


# ═══════════════════════════════════════════════════════════════════════
# REAL FLIGHT PRICE FROM LIVE SEARCH — same idea as the hotel price, for the
# round-trip flight. The deterministic route table is a realistic FLOOR; live
# search can only RAISE the per-person fare toward reality (never lower it — a
# 'from $X' hit is often a one-way/promo, so a downward move would mislead).
# Free (Tavily). Cached per route. Returns the price in the trip currency, or None.
# ═══════════════════════════════════════════════════════════════════════

_FLIGHT_PRICE_CACHE = {}


def _trip_origin(question: str) -> str:
    """Best-effort origin city for the flight. Defaults to Cairo (this app's usual origin)."""
    try:
        m = re.search(r'from\s+([A-Za-z؀-ۿ ]{3,30}?)\s+to\s+', question or '', re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r'من\s+([؀-ۿ ]{3,30}?)\s+(?:إلى|الى|ل)\s', question or '')
        if m2:
            return m2.group(1).strip()
    except Exception:
        pass
    return "Cairo"


def _real_flight_roundtrip_price(origin: str, destination: str, currency: str):
    """Live round-trip ECONOMY price PER PERSON for origin→destination, in the trip currency,
    via Tavily search. Pools several query variants, trims outliers, returns the median. Cached.
    Returns None when search finds no price (caller then keeps the deterministic route price)."""
    try:
        fx = _USD_FX.get((currency or '$').strip())
        if not fx:
            return None
        o = (origin or 'Cairo').split(',')[0].strip()
        d = (destination or '').split(',')[0].strip()
        if not d:
            return None
        ckey = (o.lower(), d.lower(), (currency or '$').strip())
        if ckey in _FLIGHT_PRICE_CACHE:
            return _FLIGHT_PRICE_CACHE[ckey]
        queries = [
            f"{o} to {d} round trip flight price economy",
            f"flight ticket {o} to {d} price round trip",
            f"رحلة طيران {o} {d} ذهاب وعودة سعر التذكرة",
        ]
        pooled = []
        for q in queries:
            try:
                pooled += _parse_night_prices_usd(_tavily_text(search_tool.invoke(q.strip())), 40, 6000)
            except Exception:
                continue
            if len(pooled) >= 6:
                break
        if not pooled:
            _FLIGHT_PRICE_CACHE[ckey] = None
            return None
        pooled.sort()
        median_usd = pooled[len(pooled) // 2]
        core = [p for p in pooled if 0.5 * median_usd <= p <= 2.5 * median_usd] or pooled
        core.sort()
        typical_usd = core[len(core) // 2]
        val = typical_usd * fx
        _FLIGHT_PRICE_CACHE[ckey] = val
        return val
    except Exception:
        return None


def _budget_hotel_line_value(draft: str):
    """Best-effort amount shown on the hotel/accommodation line of the budget table.
    Returns None when unsure."""
    try:
        for pat in [
            r'(?:تكلفة\s*الإقامة|الإقامة|الفندق)[^\n|]*\|\s*(?:\$|€|£)?\s*([\d][\d,]{2,})',
            r'(?:Hotel|Accommodation)[^\n|]*\|\s*(?:\$|€|£)?\s*([\d][\d,]{2,})',
            r'(?:تكلفة\s*الإقامة|الإقامة)\b[^\n\d]{0,20}([\d][\d,]{2,})',
        ]:
            m = re.search(pat, draft, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(',', ''))
                if val >= 500:
                    return val
    except Exception:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC BUDGET — the numbers and the table are built IN CODE, never
# left to the LLM. The LLM only CHOOSES the hotel/flight/experiences and writes
# the prose; this module reads those chosen options back out, does ALL the math
# in Python, and rebuilds the budget table so the figures can NEVER be
# inconsistent or wrong-arithmetic again.
# ═══════════════════════════════════════════════════════════════════════

def _extract_budget_amount(draft: str, labels: List[str]):
    """Amount on a budget row whose label matches any of `labels`. None if not found."""
    for lab in labels:
        for pat in [
            lab + r'[^\n|]*\|\s*(?:\$|€|£)?\s*(-?[\d][\d,]{2,})',   # markdown table cell
            lab + r'\b[^\n\d]{0,25}(-?[\d][\d,]{2,})',             # "label ... amount"
        ]:
            m = re.search(pat, draft, re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1).replace(',', ''))
                    if abs(v) >= 100:
                        return v
                except ValueError:
                    pass
    return None


def _extract_transport_total(draft: str, flight_notes, num_travelers):
    """Total transport cost for ALL travelers. Prefers the transport table's
    '× N = AMOUNT' (recommended option), then the budget's transport row, then
    cheapest flight × travelers. None if nothing reliable is found."""
    m = re.search(r'×\s*\d+\s*=\s*(?:\$|€|£)?\s*([\d][\d,]{2,})', draft)
    if m:
        try:
            v = float(m.group(1).replace(',', ''))
            if v >= 100:
                return v
        except ValueError:
            pass
    v = _extract_budget_amount(draft, ['تكلفة النقل', 'النقل', 'المواصلات', 'Transport'])
    if v:
        return v
    p, _ = _cheapest_unit_price(flight_notes, 'transport')
    if p:
        return p * max(int(num_travelers or 1), 1)
    return None


def _extract_activity_costs(draft: str):
    """Extract the per-activity GROUP costs from the activities section of the plan.
    Returns (costs: list[float], detail: str) where detail is 'a + b + c …'. Empty when the
    section or costs can't be found — so the caller can fall back to the plain total."""
    try:
        if not draft:
            return [], ""
        m = re.search(r'(?:^|\n)[ \t]*#{1,4}[^\n]*(?:الأنشطة|التجارب|Activities|Experiences)[^\n]*',
                      draft, re.IGNORECASE)
        if not m:
            return [], ""
        rest = draft[m.end():]
        nxt = re.search(r'\n[ \t]*#{1,4}[ \t]+\S', rest)
        section = rest[:nxt.start()] if nxt else rest
        costs = []
        for cm in re.finditer(
                r'(?:التكلفة|السعر|Cost|Price)[:\s]*([\d][\d,]{1,7}(?:\.\d+)?)\s*'
                r'(?:ج\.م|EGP|\$|€|£|AED|د\.إ|SAR|ر\.س)', section, re.IGNORECASE):
            try:
                v = float(cm.group(1).replace(',', ''))
                if v >= 50:
                    costs.append(v)
            except ValueError:
                pass
        if not costs:
            return [], ""
        detail = " + ".join(f"{int(round(c)):,}" for c in costs)
        return costs, detail
    except Exception:
        return [], ""


def _build_authoritative_budget(transport, hotel, activities, food, num_nights,
                                user_budget, currency, num_travelers=1,
                                activities_detail="") -> str:
    """Build the ONE budget section with the math done in code — single source of truth.
    Includes a التفاصيل (breakdown) column so it reads as the detailed budget."""
    total = transport + hotel + activities + food
    remaining = user_budget - total
    rem = _fmt_money(remaining, currency) if remaining >= 0 else "-" + _fmt_money(abs(remaining), currency)
    nights = max(int(num_nights or 1), 1)
    trav = max(int(num_travelers or 1), 1)
    night = hotel / nights
    pp = transport / trav
    act_detail = activities_detail if activities_detail else "—"
    rows = [
        "## 💰 الميزانية الإجمالية",
        "",
        "| البند | التكلفة | التفاصيل |",
        "|---|---|---|",
        f"| النقل (كل المسافرين) | {_fmt_money(transport, currency)} | {_fmt_money(pp, currency)} × {trav} |",
        f"| الإقامة ({nights} ليالٍ) | {_fmt_money(hotel, currency)} | {_fmt_money(night, currency)} × {nights} |",
        f"| الأنشطة والتجارب | {_fmt_money(activities, currency)} | {act_detail} |",
    ]
    if food and food > 0:
        rows.append(f"| الطعام (مطاعم خارجية) | {_fmt_money(food, currency)} | — |")
    else:
        rows.append("| الطعام | 0 | مشمول في الفندق |")
    rows += [
        f"| **الإجمالي الكلي** | **{_fmt_money(total, currency)}** | — |",
        f"| الميزانية المتاحة | {_fmt_money(user_budget, currency)} | — |",
        f"| الميزانية المتبقية | {rem} | — |",
    ]
    return "\n".join(rows)


def _compute_food_from_itinerary(draft: str, num_travelers):
    """Sum every PAID outside-restaurant meal scheduled in the day-by-day plan.
    The itinerary lists meals like 'العشاء: ... مطعم X (300-400 ج.م للشخص)'. Returns
    (total_EGP, found): per-person prices × travelers, a range taking its midpoint.
    Lines whose meal is covered by the hotel board are skipped, so food=0 stays 0 only
    when there is genuinely no paid outside meal. This kills the 'food = 0 while the plan
    lists restaurants' contradiction by computing food from the plan itself."""
    try:
        trav = max(int(num_travelers or 1), 1)
        meal_kw = ('عشاء', 'غداء', 'فطور', 'إفطار', 'افطار', 'مطعم',
                   'dinner', 'lunch', 'breakfast', 'restaurant')
        incl_kw = ('مشمول', 'included', 'ضمن الفندق', 'في الفندق', 'بوفيه الفندق', 'بالفندق')
        total = 0.0
        found = False
        for line in draft.splitlines():
            if not any(k in line for k in meal_kw):
                continue
            if any(k in line for k in incl_kw):
                continue  # covered by the hotel's own board — no extra cost
            # per-person RANGE: "300-400 ج.م للشخص"
            m = re.search(r'([\d,]{2,})\s*(?:-|–|—|إلى|to)\s*([\d,]{2,})\s*'
                          r'(?:ج\.م|EGP|جنيه)?\s*(?:/?\s*(?:للشخص|للفرد|per\s*person|pp))',
                          line, re.IGNORECASE)
            if m:
                lo = float(m.group(1).replace(',', '')); hi = float(m.group(2).replace(',', ''))
                total += (lo + hi) / 2.0 * trav; found = True; continue
            # per-person SINGLE: "250 ج.م للشخص"
            m2 = re.search(r'([\d,]{2,})\s*(?:ج\.م|EGP|جنيه)?\s*'
                           r'(?:/?\s*(?:للشخص|للفرد|per\s*person|pp))', line, re.IGNORECASE)
            if m2:
                total += float(m2.group(1).replace(',', '')) * trav; found = True
        return int(round(total)), found
    except Exception:
        return 0, False


def _rebuild_budget_deterministic(draft, user_budget, num_nights, num_travelers,
                                  currency, travel_style, hotel_notes, flight_notes,
                                  transport_override=None):
    """Rebuild the budget from the plan's OWN chosen options, with the math in code.
    Returns (new_draft, applied: bool, reason: str). SAFE: applies only when the hotel
    and transport figures are both found AND the recomputed total stays within budget;
    otherwise returns the draft unchanged so it can never make things worse."""
    try:
        if not user_budget:
            return draft, False, "no budget"
        nights = max(int(num_nights or 1), 1)
        night_price = _recommended_hotel_night_price(draft, hotel_notes, travel_style)
        transport = transport_override if transport_override else _extract_transport_total(draft, flight_notes, num_travelers)
        activities = _extract_budget_amount(draft, ['الأنشطة والتجارب', 'الأنشطة', 'التجارب', 'Activities']) or 0
        # Activities breakdown FROM THE ITINERARY — show each activity's cost in the التفاصيل
        # column AND recompute the total from the real sum, so the breakdown can never disagree
        # with the figure (fixes the '14,400 shown but parts add to 12,600' LLM inconsistency).
        _act_costs, _act_detail = _extract_activity_costs(draft)
        _act_sum = sum(_act_costs)
        if _act_costs and (not activities or 0.5 * activities <= _act_sum <= 1.6 * activities):
            activities = _act_sum
        else:
            _act_detail = ""   # can't show a trustworthy breakdown → keep the plain total
        # Food is computed FROM THE ITINERARY (the real paid meals), not trusted from the
        # LLM's budget row — that's what was lying ("0 / included" while restaurants were scheduled).
        food_itin, has_meals = _compute_food_from_itinerary(draft, num_travelers)
        food = food_itin if has_meals else (_extract_budget_amount(draft, ['الطعام', 'المأكولات', 'الوجبات', 'Food', 'Dining']) or 0)
        if not night_price or not transport:
            return draft, False, "low confidence (hotel or transport not found)"
        hotel = night_price * nights
        total = transport + hotel + activities + food
        if total > user_budget:
            # Recommended hotel × nights doesn't fit — that's a genuine over-budget case,
            # handled elsewhere. Don't force an over-budget display here.
            return draft, False, f"authoritative total {int(total):,} over budget {int(user_budget):,}"
        new_section = _build_authoritative_budget(transport, hotel, activities, food,
                                                  nights, user_budget, currency, num_travelers,
                                                  activities_detail=_act_detail)
        new_draft = _replace_budget_section(draft, new_section)
        reason = (f"transport={int(transport):,}, hotel={int(night_price):,}×{nights}={int(hotel):,}, "
                  f"activities={int(activities):,}, food={int(food):,}, total={int(total):,}, "
                  f"remaining={int(user_budget-total):,}")
        return new_draft, True, reason
    except Exception as e:
        return draft, False, f"error: {e}"


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC TRANSPORT — the transport mode, routing and price are chosen
# IN CODE from a curated table of realistic Egyptian routes, so the LLM can't
# hallucinate (e.g. a 6-hour "15,000/person" private car to Dahab). Rules:
#   • a destination with NO airport → fly to the nearest airport + private transfer
#   • always round trip (رايح جاي)
#   • luxury / mid-range → prefer flying over a long car ride
# These are deterministic estimates (EGP); a real flights API can replace the
# numbers later. Only applies to EGP trips on known Egyptian routes — otherwise
# it returns None and the LLM's transport is kept (safe fallback).
# ═══════════════════════════════════════════════════════════════════════

_CITY_SYN = {
    "دهب": "dahab", "dahab": "dahab",
    "شرم الشيخ": "sharm el sheikh", "شرم": "sharm el sheikh",
    "sharm el sheikh": "sharm el sheikh", "sharm": "sharm el sheikh", "sharm el-sheikh": "sharm el sheikh",
    "الغردقة": "hurghada", "غردقة": "hurghada", "hurghada": "hurghada",
    "سهل حشيش": "sahl hasheesh", "sahl hasheesh": "sahl hasheesh",
    "الجونة": "el gouna", "الجونه": "el gouna", "el gouna": "el gouna", "gouna": "el gouna", "elgouna": "el gouna",
    "سوما باي": "soma bay", "soma bay": "soma bay",
    "مكادي": "makadi", "makadi": "makadi", "makadi bay": "makadi",
    "سفاجا": "safaga", "safaga": "safaga",
    "مرسى علم": "marsa alam", "marsa alam": "marsa alam",
    "الأقصر": "luxor", "اقصر": "luxor", "luxor": "luxor",
    "أسوان": "aswan", "اسوان": "aswan", "aswan": "aswan",
    "القاهرة": "cairo", "قاهرة": "cairo", "cairo": "cairo",
    "الإسكندرية": "alexandria", "اسكندرية": "alexandria", "alexandria": "alexandria",
    "العين السخنة": "ain sokhna", "العين السخنه": "ain sokhna", "السخنة": "ain sokhna", "ain sokhna": "ain sokhna", "sokhna": "ain sokhna",
    "طابا": "taba", "taba": "taba",
    "نويبع": "nuweiba", "nuweiba": "nuweiba",
    "رأس سدر": "ras sudr", "راس سدر": "ras sudr", "ras sudr": "ras sudr",
}

_AR_NAME = {
    "sharm el sheikh": "شرم الشيخ", "hurghada": "الغردقة", "luxor": "الأقصر",
    "aswan": "أسوان", "marsa alam": "مرسى علم", "cairo": "القاهرة",
    "alexandria": "الإسكندرية", "dahab": "دهب", "el gouna": "الجونة",
    "sahl hasheesh": "سهل حشيش", "soma bay": "سوما باي", "makadi": "مكادي",
    "safaga": "سفاجا", "taba": "طابا", "nuweiba": "نويبع",
    "ain sokhna": "العين السخنة", "ras sudr": "رأس سدر",
}

# destination with NO airport → (nearest airport city, one-way transfer per car EGP, hours)
_NO_AIRPORT_DEST = {
    "dahab": ("sharm el sheikh", 2000, 1.5),
    "nuweiba": ("sharm el sheikh", 2500, 2.5),
    "taba": ("sharm el sheikh", 3500, 3.0),
    "el gouna": ("hurghada", 800, 0.5),
    "sahl hasheesh": ("hurghada", 700, 0.5),
    "soma bay": ("hurghada", 1000, 1.0),
    "makadi": ("hurghada", 900, 0.75),
    "safaga": ("hurghada", 1200, 1.0),
    "ain sokhna": ("cairo", 2000, 2.0),      # drivable from Cairo — no flight
    "ras sudr": ("cairo", 2500, 2.5),
}

# airport city reachable by flight from Cairo → round-trip per-person EGP by style
_DOMESTIC_RT = {
    "sharm el sheikh": {"budget": 6000, "mid-range": 9000, "luxury": 12000},
    "hurghada":        {"budget": 6000, "mid-range": 9000, "luxury": 11000},
    "luxor":           {"budget": 5000, "mid-range": 7500, "luxury": 10000},
    "aswan":           {"budget": 6000, "mid-range": 8500, "luxury": 11000},
    "marsa alam":      {"budget": 7000, "mid-range": 10000, "luxury": 13000},
}


def _norm_city(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.split(r'[،,]', s)[0].strip()         # drop country suffix
    s = re.sub(r'\s+', ' ', s)
    if s in _CITY_SYN:
        return _CITY_SYN[s]
    for k, v in _CITY_SYN.items():
        if k and k in s:
            return v
    return s


def _style_key(style: str) -> str:
    s = (style or "").lower()
    if any(k in s for k in ['lux', 'luxury', 'فاخر']):
        return 'luxury'
    if any(k in s for k in ['budget', 'economy', 'اقتصادي', 'رخيص', 'محدود']):
        return 'budget'
    return 'mid-range'


def _ar_city(c: str) -> str:
    return _AR_NAME.get(c, (c or "").title())


def _trans_result(mode, hours, total, trav, note, reason):
    trav = max(int(trav or 1), 1)
    return {"mode": mode, "hours": hours, "total": int(round(total)),
            "per_person": int(round(total / trav)), "trav": trav,
            "note": note, "reason": reason}


def _egypt_domestic_transport(question, destination, num_travelers, travel_style):
    """Deterministic transport for Egyptian DOMESTIC routes (EGP). Returns a dict, or None
    when the route isn't a known Egyptian one — so the caller can fall through to international."""
    try:
        trav = max(int(num_travelers or 1), 1)
        style = _style_key(travel_style)
        to_city = _norm_city(destination)
        if not to_city or to_city not in _CITY_SYN.values():
            mto = re.search(r'\bto\s+(.+?)[\.\n]', question or '', re.IGNORECASE)
            if mto:
                to_city = _norm_city(mto.group(1))
        mfrom = re.search(r'from\s+(.+?)\s+to\s+', question or '', re.IGNORECASE)
        from_city = _norm_city(mfrom.group(1)) if mfrom else 'cairo'
        if from_city not in _CITY_SYN.values():
            from_city = 'cairo'
        if not to_city or to_city == from_city:
            return None

        # Case A — destination has NO airport → fly to nearest airport + transfer
        if to_city in _NO_AIRPORT_DEST:
            airport, xfer_oneway, xfer_hrs = _NO_AIRPORT_DEST[to_city]
            cars = max((trav + 2) // 3, 1)              # ~3 people per private car
            if airport == 'cairo':                      # drivable from Cairo — no flight
                total = xfer_oneway * 2 * cars
                return _trans_result(
                    mode=f"نقل خاص (سيارة فاخرة) من {_ar_city(from_city)} إلى {_ar_city(to_city)}",
                    hours=f"~{xfer_hrs} ساعة", total=total, trav=trav,
                    note="رايح جاي (ذهاب وعودة)",
                    reason=f"{_ar_city(to_city)} قريبة وتُوصل بالسيارة مباشرة، فالنقل الخاص أريح وأوفر من الطيران")
            flight_pp = _DOMESTIC_RT.get(airport, {}).get(style)
            if not flight_pp:
                return None
            total = flight_pp * trav + xfer_oneway * 2 * cars
            return _trans_result(
                mode=f"طيران {_ar_city(from_city)} ← {_ar_city(airport)} + نقل خاص إلى {_ar_city(to_city)}",
                hours=f"طيران ~1 ساعة + نقل ~{xfer_hrs} ساعة", total=total, trav=trav,
                note="رايح جاي (ذهاب وعودة)",
                reason=(f"{_ar_city(to_city)} ملهاش مطار — فالأفضل الطيران لأقرب مطار "
                        f"({_ar_city(airport)}) ثم نقل خاص، وده أريح وأسرع بكتير من 6-8 ساعات بالسيارة"))

        # Case B — destination has its own airport → direct flight
        if to_city in _DOMESTIC_RT:
            flight_pp = _DOMESTIC_RT[to_city].get(style)
            if not flight_pp:
                return None
            total = flight_pp * trav
            return _trans_result(
                mode=f"طيران {_ar_city(from_city)} ← {_ar_city(to_city)}",
                hours="~1 ساعة", total=total, trav=trav,
                note="رايح جاي (ذهاب وعودة)",
                reason="رحلة طيران مباشرة ومريحة تناسب أسلوب السفر")
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC INTERNATIONAL FLIGHTS — round-trip ECONOMY prices from Egypt,
# by region & style, so international transport can't be hallucinated and is
# ALWAYS round trip (رايح وجاي). Base prices are USD per person round trip;
# converted to the trip currency. "luxury" = best economy / premium economy
# (a flight is transit — the luxury goes into the stay, per the agreed rule).
# ═══════════════════════════════════════════════════════════════════════

_USD_FX = {'$': 1.0, 'ج.م': 50.0, 'EGP': 50.0, '€': 0.92, '£': 0.79, 'ر.س': 3.75, 'د.إ': 3.67}

_INTL_RT_USD = {  # region -> style -> USD round-trip per person (economy)
    'gulf':          {'budget': 200, 'mid-range': 320, 'luxury': 480},
    'turkey':        {'budget': 200, 'mid-range': 300, 'luxury': 430},
    'levant':        {'budget': 180, 'mid-range': 280, 'luxury': 400},
    'europe':        {'budget': 280, 'mid-range': 450, 'luxury': 650},
    'north_africa':  {'budget': 220, 'mid-range': 340, 'luxury': 480},
    'east_africa':   {'budget': 260, 'mid-range': 400, 'luxury': 560},
    'sea':           {'budget': 450, 'mid-range': 680, 'luxury': 900},
    'south_asia':    {'budget': 350, 'mid-range': 540, 'luxury': 740},
    'east_asia':     {'budget': 560, 'mid-range': 820, 'luxury': 1100},
    'north_america': {'budget': 620, 'mid-range': 950, 'luxury': 1300},
    'oceania':       {'budget': 900, 'mid-range': 1300, 'luxury': 1700},
}

_INTL_HOURS = {
    'gulf': '~3 ساعات / ~3 hrs', 'turkey': '~2 ساعة / ~2 hrs', 'levant': '~1.5 ساعة / ~1.5 hrs',
    'europe': '~4-5 ساعات / ~4-5 hrs', 'north_africa': '~5 ساعات / ~5 hrs',
    'east_africa': '~4 ساعات / ~4 hrs', 'sea': '~9 ساعات / ~9 hrs', 'south_asia': '~6 ساعات / ~6 hrs',
    'east_asia': '~11 ساعة / ~11 hrs', 'north_america': '~12 ساعة / ~12 hrs', 'oceania': '~16 ساعة / ~16 hrs',
}

# keyword -> region. Latin keys are matched on word boundaries; Arabic keys by substring.
_INTL_REGION = {
    'uae': 'gulf', 'dubai': 'gulf', 'abu dhabi': 'gulf', 'sharjah': 'gulf',
    'الإمارات': 'gulf', 'الامارات': 'gulf', 'دبي': 'gulf', 'أبو ظبي': 'gulf', 'ابوظبي': 'gulf', 'الشارقة': 'gulf',
    'saudi': 'gulf', 'riyadh': 'gulf', 'jeddah': 'gulf', 'السعودية': 'gulf', 'الرياض': 'gulf', 'جدة': 'gulf', 'مكة': 'gulf', 'المدينة': 'gulf',
    'qatar': 'gulf', 'doha': 'gulf', 'قطر': 'gulf', 'الدوحة': 'gulf',
    'kuwait': 'gulf', 'الكويت': 'gulf', 'bahrain': 'gulf', 'البحرين': 'gulf',
    'muscat': 'gulf', 'مسقط': 'gulf', 'سلطنة عمان': 'gulf',
    'turkey': 'turkey', 'istanbul': 'turkey', 'antalya': 'turkey', 'تركيا': 'turkey', 'اسطنبول': 'turkey', 'إسطنبول': 'turkey', 'أنطاليا': 'turkey', 'انطاليا': 'turkey',
    'jordan': 'levant', 'amman': 'levant', 'الأردن': 'levant', 'الاردن': 'levant', 'عمان': 'levant', 'lebanon': 'levant', 'beirut': 'levant', 'لبنان': 'levant', 'بيروت': 'levant',
    'spain': 'europe', 'madrid': 'europe', 'barcelona': 'europe', 'seville': 'europe', 'إسبانيا': 'europe', 'اسبانيا': 'europe', 'مدريد': 'europe', 'برشلونة': 'europe', 'إشبيلية': 'europe',
    'italy': 'europe', 'rome': 'europe', 'milan': 'europe', 'venice': 'europe', 'إيطاليا': 'europe', 'ايطاليا': 'europe', 'روما': 'europe', 'ميلان': 'europe', 'البندقية': 'europe',
    'france': 'europe', 'paris': 'europe', 'فرنسا': 'europe', 'باريس': 'europe',
    'london': 'europe', 'england': 'europe', 'britain': 'europe', 'بريطانيا': 'europe', 'لندن': 'europe', 'إنجلترا': 'europe',
    'germany': 'europe', 'berlin': 'europe', 'munich': 'europe', 'ألمانيا': 'europe', 'المانيا': 'europe', 'برلين': 'europe', 'ميونخ': 'europe',
    'switzerland': 'europe', 'zurich': 'europe', 'geneva': 'europe', 'سويسرا': 'europe', 'زيورخ': 'europe', 'جنيف': 'europe',
    'greece': 'europe', 'athens': 'europe', 'اليونان': 'europe', 'أثينا': 'europe', 'اثينا': 'europe',
    'netherlands': 'europe', 'amsterdam': 'europe', 'هولندا': 'europe', 'أمستردام': 'europe',
    'austria': 'europe', 'vienna': 'europe', 'النمسا': 'europe', 'فيينا': 'europe',
    'czech': 'europe', 'prague': 'europe', 'تشيك': 'europe', 'براغ': 'europe',
    'portugal': 'europe', 'lisbon': 'europe', 'البرتغال': 'europe', 'لشبونة': 'europe',
    'hungary': 'europe', 'budapest': 'europe', 'المجر': 'europe', 'بودابست': 'europe',
    'morocco': 'north_africa', 'marrakech': 'north_africa', 'casablanca': 'north_africa', 'المغرب': 'north_africa', 'مراكش': 'north_africa', 'الدار البيضاء': 'north_africa',
    'tunisia': 'north_africa', 'تونس': 'north_africa', 'algeria': 'north_africa', 'الجزائر': 'north_africa',
    'kenya': 'east_africa', 'nairobi': 'east_africa', 'كينيا': 'east_africa', 'tanzania': 'east_africa', 'zanzibar': 'east_africa', 'تنزانيا': 'east_africa', 'زنجبار': 'east_africa',
    'thailand': 'sea', 'bangkok': 'sea', 'phuket': 'sea', 'تايلاند': 'sea', 'بانكوك': 'sea', 'بوكيت': 'sea',
    'malaysia': 'sea', 'kuala lumpur': 'sea', 'ماليزيا': 'sea', 'كوالالمبور': 'sea',
    'indonesia': 'sea', 'bali': 'sea', 'jakarta': 'sea', 'إندونيسيا': 'sea', 'اندونيسيا': 'sea', 'بالي': 'sea',
    'singapore': 'sea', 'سنغافورة': 'sea', 'vietnam': 'sea', 'فيتنام': 'sea',
    'india': 'south_asia', 'delhi': 'south_asia', 'mumbai': 'south_asia', 'الهند': 'south_asia',
    'maldives': 'south_asia', 'المالديف': 'south_asia', 'sri lanka': 'south_asia', 'سريلانكا': 'south_asia',
    'japan': 'east_asia', 'tokyo': 'east_asia', 'اليابان': 'east_asia', 'طوكيو': 'east_asia',
    'china': 'east_asia', 'الصين': 'east_asia', 'korea': 'east_asia', 'seoul': 'east_asia', 'كوريا': 'east_asia', 'سيول': 'east_asia',
    'usa': 'north_america', 'america': 'north_america', 'new york': 'north_america', 'washington': 'north_america', 'أمريكا': 'north_america', 'امريكا': 'north_america', 'الولايات المتحدة': 'north_america', 'نيويورك': 'north_america',
    'canada': 'north_america', 'toronto': 'north_america', 'كندا': 'north_america', 'تورنتو': 'north_america',
    'australia': 'oceania', 'sydney': 'oceania', 'melbourne': 'oceania', 'أستراليا': 'oceania', 'استراليا': 'oceania', 'سيدني': 'oceania',
}


def _intl_region(text: str):
    """Detect the destination region for international flight pricing. Latin keys match on word
    boundaries (so 'oman' never matches inside 'romania'); Arabic keys match by substring."""
    if not text:
        return None
    low = text.lower()
    # longer keys first so 'abu dhabi' / 'kuala lumpur' win over short fragments
    for key in sorted(_INTL_REGION, key=len, reverse=True):
        if key.isascii():
            if re.search(r'\b' + re.escape(key) + r'\b', low):
                return _INTL_REGION[key]
        else:
            if key in text:
                return _INTL_REGION[key]
    return None


def _international_transport(question, destination, num_travelers, travel_style, currency):
    """Deterministic round-trip ECONOMY flight for an international trip from Egypt.
    Returns a transport dict, or None when the region/currency isn't covered (keep the LLM)."""
    try:
        sym = (currency or '$').strip()
        fx = _USD_FX.get(sym)
        if not fx:
            return None
        region = _intl_region(f"{destination or ''}\n{question or ''}")
        if not region:
            return None
        style = _style_key(travel_style)
        usd_pp = _INTL_RT_USD[region][style]
        trav = max(int(num_travelers or 1), 1)
        pp = usd_pp * fx
        total = pp * trav
        dest_disp = (destination or '').split(',')[0].strip() or region.replace('_', ' ').title()
        return _trans_result(
            mode=f"طيران اقتصادي (رايح وجاي) إلى {dest_disp} / Economy flight (round trip) to {dest_disp}",
            hours=_INTL_HOURS.get(region, ''), total=total, trav=trav,
            note="رحلة رايح وجاي — درجة اقتصادية / round trip, economy",
            reason=("أفضل رحلة اقتصادية (مباشرة أو بأقل توقف) تناسب الميزانية؛ السعر ذهاب وعودة للفرد — "
                    "الطيران وسيلة انتقال، والفخامة في الإقامة / best-value round-trip economy; the flight "
                    "is transit, the luxury goes into the stay"))
    except Exception:
        return None


def _compute_transport(question, destination, num_travelers, travel_style, currency):
    """Deterministic transport orchestrator. Tries the Egyptian DOMESTIC table first (EGP trips),
    then the INTERNATIONAL round-trip table. Returns a dict, or None to keep the LLM's transport."""
    try:
        is_egp = ('ج.م' in (currency or '') or 'EGP' in (currency or '').upper())
        if is_egp:
            dom = _egypt_domestic_transport(question, destination, num_travelers, travel_style)
            if dom:
                return dom
        return _international_transport(question, destination, num_travelers, travel_style, currency)
    except Exception:
        return None


def _build_transport_section(info, currency) -> str:
    pp = _fmt_money(info["per_person"], currency)
    tot = _fmt_money(info["total"], currency)
    is_egp = ('ج.م' in (currency or '') or 'EGP' in (currency or '').upper())
    badge_ar = (f"\n> 🔎 تم التحقق من سعر الطيران مقابل بحث مباشر (~{pp}/فرد ذهاب وعودة) "
                "— flight price verified against live search." if info.get("_price_verified") else "")
    badge_en = (f"\n> 🔎 Flight price verified against live search (~{pp}/person round trip)."
                if info.get("_price_verified") else "")
    if is_egp:
        return "\n".join([
            "## ✈️ النقل",
            "",
            "| وسيلة النقل | المدة | التكلفة/فرد (رايح وجاي) | الإجمالي (عدد المسافرين) | الملاحظات |",
            "|---|---|---|---|---|",
            f"| {info['mode']} | {info['hours']} | {pp} | {pp} × {info['trav']} = {tot} | {info['note']} |",
            "",
            f"🏆 توصية الـAI: {info['mode']} — {info['reason']}." + badge_ar,
        ])
    # non-EGP (English-leaning) draft → English headers
    return "\n".join([
        "## ✈️ Transport",
        "",
        "| Transport | Duration | Cost/Person (round trip) | Total (all travelers) | Notes |",
        "|---|---|---|---|---|",
        f"| {info['mode']} | {info['hours']} | {pp} | {pp} × {info['trav']} = {tot} | {info['note']} |",
        "",
        f"🏆 AI Recommendation: {info['mode']} — {info['reason']}." + badge_en,
    ])


def _enforce_transport_math(resp: str, num_travelers, currency) -> tuple:
    """UNIVERSAL GUARANTEE — in the transport table, Total MUST equal (per-person × travelers).
    Anchors on the per-person round-trip cost (the figure the user verifies) and recomputes the
    total cell. Skips any row that already shows the multiplication (e.g. '7,500 × 2 = 15,000'),
    so the deterministic transport section is left untouched. This catches every destination the
    curated route tables don't cover, so '7,500/person but 12,000 for two' can never appear again.
    Returns (resp, changed)."""
    try:
        trav = max(int(num_travelers or 1), 1)
        if trav < 2 or not resp:
            return resp, False
        lines = resp.splitlines()
        # locate the transport section header
        start = None
        for i, ln in enumerate(lines):
            if re.match(r'\s*#{1,4}\s', ln) and re.search(
                    r'(وسيلة النقل|المواصلات|النقل|Transport|Flights)', ln, re.IGNORECASE):
                start = i
                break
        if start is None:
            return resp, False
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if re.match(r'\s*#{1,4}\s', lines[j]):
                end = j
                break
        # find the header row and the per-person / total column indices
        pp_col = tot_col = hdr_idx = None
        for k in range(start, end):
            if lines[k].count('|') < 3:
                continue
            cells = [c.strip().lower() for c in lines[k].strip().strip('|').split('|')]
            for ci, c in enumerate(cells):
                if pp_col is None and ('فرد' in c or 'per person' in c or 'per-person' in c
                                       or '/person' in c or 'للفرد' in c):
                    pp_col = ci
                if tot_col is None and ('إجمالي' in c or 'الاجمالي' in c or 'المجموع' in c
                                        or 'total' in c or 'all travelers' in c):
                    tot_col = ci
            if pp_col is not None and tot_col is not None:
                hdr_idx = k
                break
        if hdr_idx is None or pp_col is None or tot_col is None or pp_col == tot_col:
            return resp, False
        num_re = re.compile(r'([\d][\d,]{2,}(?:\.\d+)?)')
        changed = False
        for k in range(hdr_idx + 1, end):
            if lines[k].count('|') < 3:
                continue
            cells = [c.strip() for c in lines[k].strip().strip('|').split('|')]
            if len(cells) <= max(pp_col, tot_col):
                continue
            if set(''.join(cells)) <= set('-: *'):
                continue
            tot_cell = cells[tot_col]
            if '×' in tot_cell or '*' in tot_cell or ' x ' in tot_cell.lower():
                continue                               # already shows pp × n = total → trust it
            mpp = num_re.search(cells[pp_col])
            mtot = num_re.search(tot_cell)
            if not mpp or not mtot:
                continue
            try:
                pp = float(mpp.group(1).replace(',', ''))
                tot = float(mtot.group(1).replace(',', ''))
            except ValueError:
                continue
            want = pp * trav
            if abs(want - tot) > max(1, want * 0.02):
                cells[tot_col] = tot_cell.replace(mtot.group(1), f"{int(round(want)):,}", 1)
                lines[k] = '| ' + ' | '.join(cells) + ' |'
                changed = True
        return "\n".join(lines), changed
    except Exception:
        return resp, False


def _replace_section(draft, header_keywords, new_section) -> str:
    """Replace a markdown section (its header line through to the next header) with
    new_section. Returns the draft unchanged if no matching header is found."""
    kw = "|".join(re.escape(k) for k in header_keywords)
    hdr = re.compile(r'(^|\n)[ \t]*#{1,4}[ \t]*[^\n]*(?:' + kw + r')[^\n]*', re.IGNORECASE)
    m = hdr.search(draft)
    if not m:
        return draft
    start = m.start() if m.start() == 0 else m.start() + 1
    rest = draft[m.end():]
    nxt = re.search(r'\n[ \t]*#{1,4}[ \t]+\S', rest)
    end = (m.end() + nxt.start()) if nxt else len(draft)
    return draft[:start].rstrip() + "\n\n" + new_section + "\n\n" + draft[end:].lstrip()


def _force_budget_fit(resp: str, flight_notes, hotel_notes, user_budget, num_nights,
                      num_travelers, max_transport_per_person, max_hotel_per_night, currency,
                      travel_style="luxury"):
    """DETERMINISTIC GUARANTEE — the final safety net.

    If the draft is still over budget, rebuild the budget section using the CHEAPEST flight +
    hotel found in research (falling back to the budget caps). The plan is NEVER shown over
    budget while a cheaper combination fits. Only when even the cheapest feasible combination
    exceeds the budget do we show an honest "increase the budget" notice — which is the one
    acceptable case for an over-budget result.

    Returns (new_resp, status): status is "ok" (already fit or fixed) or "infeasible".
    """
    check = _extract_total_from_draft(resp, user_budget)
    if not check.get("is_over_budget"):
        return resp, "ok"
    if not user_budget:
        return resp, "ok"

    num_travelers = max(int(num_travelers or 1), 1)
    num_nights = max(int(num_nights or 1), 1)

    # ── cheapest realistic per-unit from research ──
    cheap_fp, fp_label = _cheapest_unit_price(flight_notes, "transport")
    cheap_hn, hn_label = _cheapest_unit_price(hotel_notes, "hotel")

    cur = currency or "$"
    style = (travel_style or "").lower()
    is_budget_style = any(k in style for k in ['budget', 'اقتصادي', 'رخيص', 'economy'])

    # ── STEP 1 — MINIMIZE TRANSPORT (the flight is just transit) ──
    # The cheapest REAL flight found in research is the achievable floor. The cap is only a
    # fallback when research gave no price — we must not invent an unbookable cheap flight.
    tp_per_person = cheap_fp if cheap_fp else (max_transport_per_person or 0)
    transport_total = tp_per_person * num_travelers

    # ── FEASIBILITY — cheapest flight + cheapest affordable hotel must fit ──
    floor_hn = cheap_hn if cheap_hn else (max_hotel_per_night or 0)
    min_needed = transport_total + floor_hn * num_nights
    if min_needed > user_budget:
        # Even the cheapest flight + cheapest hotel overshoots → honest "increase budget" notice.
        needed = _fmt_money(min_needed, cur)
        note = (
            f"⚠️ **الميزانية غير كافية لهذه الوجهة.** حتى بأرخص رحلة طيران متاحة "
            f"({_fmt_money(tp_per_person, cur)}/شخص) وأرخص فندق مناسب ({_fmt_money(floor_hn, cur)}/ليلة)، "
            f"تبدأ التكلفة من حوالي {needed} لـ {num_travelers} مسافرين و{num_nights} ليالٍ — "
            f"وهذا يتجاوز ميزانيتك ({_fmt_money(user_budget, cur)}).\n\n"
            f"**الحلول المقترحة:** زيادة الميزانية إلى {needed} على الأقل، أو تقليل عدد الليالي/المسافرين، "
            f"أو اختيار وجهة أقرب وأقل تكلفة."
        )
        new_section = _build_budget_section(
            transport_total, floor_hn * num_nights, 0, num_nights, user_budget, cur, note)
        return _replace_budget_section(resp, new_section), "infeasible"

    # ── STEP 2 — SPEND THE SAVED MONEY ON THE STAY ──
    hn_label = None
    if is_budget_style:
        # Budget traveler: keep it lean; a large leftover is fine for them.
        hn_per_night = floor_hn
        hotel_total = hn_per_night * num_nights
        activities_total = max(int(user_budget * 0.08), 0)
        # never exceed budget
        if transport_total + hotel_total + activities_total > user_budget:
            activities_total = max(int(user_budget - transport_total - hotel_total), 0)
        fill_note = (
            "✅ **تم ضبط الخطة لتناسب ميزانيتك** باختيار أرخص وسيلة نقل مناسبة."
        )
    else:
        # LUXURY / MID-RANGE: use ~96% of the budget — raise the HOTEL level and the
        # dining/activities, since the money saved on the flight belongs in the experience.
        usable = user_budget * 0.96 - transport_total           # hotel + activities pool
        if usable < floor_hn * num_nights:                      # safety (shouldn't happen post-feasibility)
            usable = user_budget - transport_total
        afford_per_night = int((usable * 0.70) / num_nights)    # hotel may take up to ~70% of the pool
        # Pick the BEST real hotel the pool can afford; else use the affordable ceiling itself.
        pricey_hn, hn_label = _priciest_unit_price(hotel_notes, "hotel", ceiling=max(afford_per_night, floor_hn))
        hn_per_night = pricey_hn if pricey_hn else afford_per_night
        hn_per_night = max(hn_per_night, floor_hn)               # at least the cheapest real hotel
        hotel_total = hn_per_night * num_nights
        # Everything else in the pool goes to dining + activities + outings.
        activities_total = max(int(usable - hotel_total), int(user_budget * 0.08))
        # Guarantee we stay strictly under budget (trim activities, then hotel, if rounding pushed over).
        total = transport_total + hotel_total + activities_total
        if total > user_budget:
            over = total - user_budget
            activities_total = max(activities_total - int(over), 0)
            total = transport_total + hotel_total + activities_total
            if total > user_budget:
                hotel_total = max(hotel_total - int(total - user_budget), floor_hn * num_nights)
        fill_note = (
            "✅ **تم ضبط الخطة لتناسب ميزانيتك مع الحفاظ على مستوى فاخر.** قلّلنا تكلفة الطيران "
            "لأقل خيار اقتصادي مناسب (الطيران مجرد وسيلة انتقال)، وحوّلنا الفرق لرفع مستوى الفندق "
            "والمطاعم والأنشطة — عشان تستغل ميزانيتك بالكامل في تجربة أفخم، من غير ما تتعدّى الميزانية."
        )

    # ── adjustment-source note ──
    def _clean_label(lbl):
        if not lbl:
            return ""
        lbl = re.sub(r'^[\s✅⚠️⛔]*\[[^\]]*\]\s*', '', lbl)
        return lbl.strip(" .*").strip()
    src_bits = []
    fp_clean, hn_clean = _clean_label(fp_label), _clean_label(hn_label)
    if fp_clean:
        src_bits.append(f"الطيران: {fp_clean}")
    if hn_clean:
        src_bits.append(f"الفندق: {hn_clean}")
    src = ("\n\n*" + " — ".join(src_bits) + "*") if src_bits else ""

    new_section = _build_budget_section(
        transport_total, hotel_total, activities_total, num_nights, user_budget, cur, fill_note + src)
    return _replace_budget_section(resp, new_section), "ok"


def _is_budget_heading(header_text: str) -> bool:
    """True when a markdown header line is the budget SUMMARY heading (not a daily/other one)."""
    if '💰' in header_text:
        return True
    t = re.sub(r'[#*_`💰🧾]', '', header_text).strip()
    if re.search(r'يومي|daily', t, re.IGNORECASE):     # 'الميزانية اليومية' is NOT the summary
        return False
    return bool(re.match(r'^(?:ميزانية|الميزانية)(?:\s+(?:الإجمالية|الكلية|الكاملة|التفصيلية))?\s*$', t)) \
        or bool(re.match(r'^(?:total\s+|overall\s+)?budget(?:\s+breakdown)?\s*$', t, re.IGNORECASE))


def _replace_budget_section(resp: str, new_section: str) -> str:
    """Keep EXACTLY ONE budget section. Replace the FIRST budget section with the rebuilt
    (authoritative) one, and DELETE every other budget section so the page never shows the
    budget twice. Appends the authoritative section if none exists."""
    hdr_re = re.compile(r'(?:\A|\n)([ \t]*#{1,4}[ \t]*[^\n]*)')
    spans = []                       # (start, end) of each budget section
    for m in hdr_re.finditer(resp):
        if not _is_budget_heading(m.group(1)):
            continue
        start = m.start()
        if resp[start] == '\n':
            start += 1
        rest = resp[m.end():]
        nxt = re.search(r'\n[ \t]*#{1,4}[ \t]+\S|\n[ \t]*(?:🆘|📞|🏥|🧳|🎒|⚠️)', rest)
        end = m.end() + nxt.start() if nxt else len(resp)
        spans.append((start, end))
    if not spans:
        return (resp.rstrip() + "\n\n" + new_section + "\n")
    spans.sort()
    out, cursor = [], 0
    for idx, (s, e) in enumerate(spans):
        out.append(resp[cursor:s])
        if idx == 0:                 # first budget → replace with the authoritative one
            out.append(new_section.strip() + "\n")
        # every later budget section is dropped
        cursor = e
    out.append(resp[cursor:])
    result = "".join(out)
    return re.sub(r'\n{3,}', '\n\n', result).strip() + "\n"


# ─────────────────────────────────────────────
# 6. Agent Nodes
# ─────────────────────────────────────────────

def planner_node(state: GraphState) -> GraphState:
    # Capture the user's chosen currency up front so every node renders prices in it.
    code, _sym = _currency_from_question(state.get("question", ""))
    state["user_currency"] = code
    print(f"[CODE-LEVEL] User currency: {code} ({_sym})")

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

    # ── Use structured budget data from GraphState ──
    total_budget = state.get('user_budget') or 0
    num_travelers = state.get('num_travelers', 1)
    max_transport = int(total_budget * 0.35) if total_budget else "unknown"
    max_per_person = int(max_transport / num_travelers) if total_budget and num_travelers else "unknown"

    output = _invoke_agent(agent,
        f"Find transport for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}\nTravel style: {travel_style}\n\n"
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║  BUDGET HARD LIMITS — VIOLATING THESE = PLAN REJECTION     ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
        f"TOTAL TRIP BUDGET: {int(total_budget) if total_budget else 'unknown'}\n"
        f"Number of travelers: {num_travelers}\n"
        f"MAXIMUM transport budget (35% of total): {max_transport}\n"
        f"MAXIMUM per person: {max_per_person}\n"
        f"═══════════════════════════════════════════════════════════════\n"
        f"If business/first class for {num_travelers} people exceeds {max_transport}, DO NOT recommend it.\n"
        f"Pick economy class on a good airline instead.\n"
        f"'Luxury' travel style does NOT mean business class if it blows the budget.\n"
        f"'Luxury' means: best economy seat, good airline, convenient times.\n\n"
        f"ALWAYS present exactly 3 transport options, cheapest to most expensive.\n"
        f"Every option MUST cost less than {max_transport} total for {num_travelers} people."
    )
    state["flight_notes"] = _parse_notes(output)
    # CODE-LEVEL: Inject budget-compliant option if all options are over budget
    if total_budget and max_per_person:
        destination = state['plan'].get('destination', '')
        state["flight_notes"] = _inject_budget_compliant_option(
            state["flight_notes"], max_per_person, "transport", destination
        )
    return state


def hotel_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(HOTEL_SYSTEM)
    travel_style = state['plan'].get('travel_style', 'mid-range')
    num_days = state['plan']['num_days']
    # num_days represents the NIGHTS booked: the frontend sets
    # check-out date = arrival date + num_days, so the guest stays num_days nights.
    num_nights = max(num_days, 1)

    # ── Use structured budget data from GraphState ──
    total_budget = state.get('user_budget') or 0
    num_travelers = state.get('num_travelers', 1)
    max_hotel_total = int(total_budget * 0.45) if total_budget else 0
    max_per_night = int(max_hotel_total / num_nights) if total_budget and num_nights else 0

    # ← ORCHESTRATOR PATTERN: Hotel receives flight context for transport-aware recommendations
    flight_context = ""
    if state.get('flight_notes'):
        flight_summary = chr(10).join('- ' + n for n in state['flight_notes'][:5])
        flight_context = f"\n\n══ CONTEXT FROM TRANSPORT AGENT ══\nThe traveler will arrive via:\n{flight_summary}\nConsider hotel proximity to arrival point (airport/bus station).\n══════════════════════════════════\n"

    budget_constraint = ""
    if total_budget:
        budget_constraint = (
            f"\n╔══════════════════════════════════════════════════════════════╗\n"
            f"║  BUDGET HARD LIMITS — VIOLATING THESE = PLAN REJECTION     ║\n"
            f"╚══════════════════════════════════════════════════════════════╝\n"
            f"TOTAL TRIP BUDGET: {int(total_budget)}\n"
            f"Number of travelers: {num_travelers}\n"
            f"Number of nights: {num_nights}\n"
            f"MAXIMUM hotel budget (45% of total): {max_hotel_total}\n"
            f"MAXIMUM per night: {max_per_night}\n"
            f"═══════════════════════════════════════════════════════════════\n"
            f"⚠️ ALL 3 hotel options MUST cost ≤ {max_per_night}/night.\n"
            f"If a luxury hotel costs more than {max_per_night}/night, DO NOT recommend it.\n"
            f"Pick the best hotel WITHIN budget instead.\n"
            f"'Luxury' does NOT mean exceeding the budget — it means the BEST option within {max_per_night}/night.\n"
            f"If {max_per_night}/night only gets a 4-star hotel, recommend that 4-star.\n"
            f"Tell the user: 'لو عايز فندق 5 نجوم فاخر، يُنصح بزيادة الميزانية.'\n"
        )

    output = _invoke_agent(agent,
        f"Find hotels for: {state['question']}\nDestination: {state['plan']['destination']}\nDates: {state['plan']['travel_dates']}\nNumber of nights: {num_nights}\nTravel style: {travel_style}\n{flight_context}{budget_constraint}\nIMPORTANT: The guest stays {num_nights} nights (itinerary covers {num_nights + 1} days: arrival day through check-out day). Search for REAL hotel prices per night. ALL options MUST be within the budget limit above."
    )
    state["hotel_notes"] = _parse_notes(output)
    # CODE-LEVEL: Inject budget-compliant option if all options are over budget
    if total_budget and max_per_night:
        destination = state['plan'].get('destination', '')
        state["hotel_notes"] = _inject_budget_compliant_option(
            state["hotel_notes"], max_per_night, "hotel", destination
        )
    return state


def _sanitize_visa_notes(notes: List[str]) -> List[str]:
    """Remove any note that leaked hotel / activity / restaurant / attraction / weather
    content into the visa section. The Visa tab must show ONLY visa/entry info —
    hotels are shown once in the full itinerary, activities once in the full itinerary."""
    if not notes:
        return notes

    # Keywords that indicate the line is NOT visa content (it leaked from another agent).
    forbidden = [
        # hotels / accommodation
        'فندق', 'فنادق', 'الفنادق', 'ريزورت', 'منتجع', 'نزل', 'إقامة',
        'hotel', 'resort', 'accommodation', 'ليلة', '/night', 'للّيلة', 'لليلة', 'لليلتين',
        'نجوم', 'نجمة', 'star', '⭐',
        # activities / excursions / tours
        'نشاط', 'الأنشطة', 'أنشطة', 'رحلة بحرية', 'غطس', 'غوص', 'سفاري', 'جولة', 'جولات',
        'activity', 'activities', 'excursion', 'tour', 'snorkel', 'diving', 'safari',
        # restaurants / dining / attractions
        'مطعم', 'مطاعم', 'مقهى', 'كافيه', 'شاطئ', 'معلم', 'معالم', 'زيارة',
        'restaurant', 'cafe', 'dining', 'beach', 'attraction',
        # weather (belongs to weather agent / weather tab)
        'طقس', 'حرارة', 'درجة الحرارة', 'weather', 'temperature', 'rainfall', 'humidity',
    ]
    # Words that confirm a line IS visa content — keep it even if it brushes a forbidden word.
    visa_safe = [
        'تأشير', 'تأشيرة', 'فيزا', 'جواز', 'جوازات', 'باسبور', 'دخول', 'إقامة قانونية',
        'visa', 'passport', 'entry', 'e-visa', 'visa-on-arrival', 'عند الوصول',
        'صلاحية الجواز', 'رسوم التأشيرة', 'لا حاجة لتأشيرة', 'no visa',
    ]

    cleaned = []
    for note in notes:
        low = note.lower()
        is_visa = any(k.lower() in low for k in visa_safe)
        is_forbidden = any(k.lower() in low for k in forbidden)
        # Keep a line if it's clearly visa content, OR if it doesn't look like leaked content.
        if is_visa or not is_forbidden:
            cleaned.append(note)
        else:
            print(f"[CODE-LEVEL] Stripped leaked line from visa_notes: {note[:70]}")

    # If sanitizing removed everything (the agent returned only leaked content),
    # fall back to a safe minimal message rather than an empty section.
    if not cleaned:
        cleaned = ["يرجى التأكد من متطلبات التأشيرة وصلاحية جواز السفر قبل السفر."]
    return cleaned


def visa_node(state: GraphState) -> GraphState:
    agent = _make_research_agent(VISA_SYSTEM)

    # ── Extract nationality from the question ──
    nationality = "unknown"
    for line in state['question'].split('\n'):
        line_stripped = line.strip()
        if 'nationality' in line_stripped.lower() or 'جنسي' in line_stripped:
            if ':' in line_stripped:
                nationality = line_stripped.split(':', 1)[1].strip()
            break

    destination = state['plan']['destination']
    # ── destination_country comes from the Planner (TravelPlan model) ──
    destination_country = state['plan'].get('destination_country', '').strip()

    # ── Map nationality to country name (small map — just nationalities) ──
    nationality_to_country = {
        'مصري': 'Egypt', 'egyptian': 'Egypt', 'مصرية': 'Egypt',
        'سعودي': 'Saudi Arabia', 'saudi': 'Saudi Arabia', 'سعودية': 'Saudi Arabia',
        'إماراتي': 'UAE', 'emirati': 'UAE',
        'أردني': 'Jordan', 'jordanian': 'Jordan',
        'لبناني': 'Lebanon', 'lebanese': 'Lebanon',
        'عراقي': 'Iraq', 'iraqi': 'Iraq',
        'كويتي': 'Kuwait', 'kuwaiti': 'Kuwait', 'كويتية': 'Kuwait',
        'بحريني': 'Bahrain', 'bahraini': 'Bahrain',
        'عماني': 'Oman', 'omani': 'Oman',
        'قطري': 'Qatar', 'qatari': 'Qatar',
        'تونسي': 'Tunisia', 'tunisian': 'Tunisia',
        'مغربي': 'Morocco', 'moroccan': 'Morocco',
        'جزائري': 'Algeria', 'algerian': 'Algeria',
        'سوري': 'Syria', 'syrian': 'Syria',
        'أمريكي': 'USA', 'american': 'USA',
        'بريطاني': 'UK', 'british': 'UK',
    }
    traveler_country = nationality_to_country.get(nationality.lower().strip(), '')

    # ── DETERMINISTIC: same country or different? ──
    # Normalize both to lowercase for comparison
    tc = traveler_country.lower()
    dc = destination_country.lower()
    # Handle common name variations
    is_domestic = False
    if tc and dc:
        is_domestic = (tc == dc
            or tc in dc or dc in tc  # "egypt" in "egypt" or partial matches
        )

    if is_domestic:
        state["visa_notes"] = ["لا حاجة لتأشيرة — المسافر داخل بلده."]
        return state

    # ── INTERNATIONAL → research visa requirements ──
    output = _invoke_agent(agent,
        f"""═══ VISA RESEARCH — INTERNATIONAL TRAVEL ═══
⚠️ THIS IS INTERNATIONAL TRAVEL. DO NOT say "no visa needed".

TRAVELER NATIONALITY: {nationality}
TRAVELER'S HOME COUNTRY: {traveler_country}
DESTINATION: {destination}
DESTINATION COUNTRY: {destination_country}

{traveler_country} ≠ {destination_country} → VISA RESEARCH REQUIRED.

Search for: "{nationality} passport visa to {destination_country}"

Original request: {state['question']}
═══════════════════════════════════════════════════"""
    )
    # Strip any hotel/activity/restaurant/weather content that leaked into visa research.
    state["visa_notes"] = _sanitize_visa_notes(_parse_notes(output))
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

    num_nights = max(num_days, 1)  # num_days = nights booked (check-out = arrival + num_days)
    output = _invoke_agent(agent,
        f"Calculate travel budget for: {state['question']}\nDestination: {state['plan']['destination']}\n"
        f"HOTEL NIGHTS: {num_nights} (hotel total = price/night × {num_nights} nights).\n"
        f"ITINERARY DAYS: {num_nights + 1} (arrival day through check-out day — for meals/activities planning).\n"
        f"Calculate for exactly {num_nights} nights, not more.\nTravel style: {travel_style}\n\n"
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
Duration: {max(num_days, 1)} nights ({max(num_days, 1) + 1} itinerary days: arrival day → check-out day)
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

    user_budget = state.get('user_budget') or 0
    num_travelers = state.get('num_travelers', 1)
    cur_code, cur_sym = _currency_from_question(state.get('question', '')) \
        if not state.get('user_currency') else (state['user_currency'], CURRENCY_SYMBOLS.get(state['user_currency'], '$'))
    max_transport = int(user_budget * 0.35) if user_budget else 0
    max_hotel = int(user_budget * 0.45) if user_budget else 0
    # num_days = nights booked (check-out = arrival + num_days).
    # Itinerary spans num_days + 1 dated days (arrival day → check-out day).
    num_nights = max(num_days, 1)
    itinerary_days = num_nights + 1
    num_days_for_hotel = num_nights
    max_hotel_per_night = int(max_hotel / num_days_for_hotel) if max_hotel else 0
    max_transport_per_person = int(max_transport / max(num_travelers, 1)) if max_transport else 0

    currency_directive = (
        f"\n╔══════════════════════════════════════════════════════════════╗\n"
        f"║  CURRENCY — MANDATORY: use {cur_code} ({cur_sym}) for EVERY price ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
        f"The traveler chose {cur_code}. EVERY number (flights, hotels, activities, dining, totals) MUST be in "
        f"{cur_code} and shown with the symbol '{cur_sym}'. The budget {int(user_budget):,} is in {cur_code}.\n"
        f"⚠️ Any 'ج.م' you see in the format templates is only a PLACEHOLDER — replace it with '{cur_sym}'. "
        f"Do NOT output 'ج.م' unless {cur_code} is EGP. Never mix currencies.\n"
    )

    # ══════════════════════════════════════════════════════════════
    # CODE-LEVEL FILTER: Remove over-budget options BEFORE the LLM sees them.
    # If the LLM never sees a $1000/night hotel, it CAN'T pick it.
    # ══════════════════════════════════════════════════════════════
    flight_notes = state.get('flight_notes', [])
    hotel_notes = state.get('hotel_notes', [])
    destination = state['plan'].get('destination', '')
    if user_budget:
        flight_notes = _filter_over_budget_options(
            flight_notes, max_transport_per_person, "transport"
        )
        hotel_notes = _filter_over_budget_options(
            hotel_notes, max_hotel_per_night, "hotel"
        )
        # Inject budget-compliant options if all were filtered out
        flight_notes = _inject_budget_compliant_option(
            flight_notes, max_transport_per_person, "transport", destination
        )
        hotel_notes = _inject_budget_compliant_option(
            hotel_notes, max_hotel_per_night, "hotel", destination
        )
        print(f"[CODE-LEVEL] Filtered research: {len(state.get('flight_notes',[]))} flight notes → {len(flight_notes)}, "
              f"{len(state.get('hotel_notes',[]))} hotel notes → {len(hotel_notes)}")

    resp = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

══════ CRITICAL CONSTRAINTS ══════
ITINERARY DAYS: {itinerary_days} — write EXACTLY {itinerary_days} dated days.
  • Day 1 = ARRIVAL day (check-in around noon; afternoon/evening only).
  • Days 2 to {num_nights} = FULL days.
  • Day {itinerary_days} = CHECK-OUT / departure day (morning only; check-out + transfer to airport).
HOTEL NIGHTS: {num_nights} (guest sleeps {num_nights} nights, checks out on day {itinerary_days}).
Do NOT cram check-out into the last full day — the check-out day is its own separate day.
TRAVEL STYLE: {travel_style} — ALL recommendations must match this style.
══════════════════════════════════
{currency_directive}
╔══════════════════════════════════════════════════════════════╗
║  BUDGET HARD LIMITS — YOUR PLAN WILL BE REJECTED IF EXCEEDED ║
╚══════════════════════════════════════════════════════════════╝
TOTAL BUDGET: {int(user_budget):,} {cur_sym} (ABSOLUTE MAXIMUM — plan total MUST be LESS)
Number of travelers: {num_travelers}
MAX transport (35%): {max_transport:,} (for ALL {num_travelers} travelers combined)
MAX per person transport: {max_transport_per_person:,}
MAX hotel total (45%): {max_hotel:,} (for ALL {num_days_for_hotel} nights combined)
MAX hotel per night: {max_hotel_per_night:,}

⚠️ CRITICAL: Pick options WITHIN these limits. Over-budget options have been REMOVED from research below.
If ALL options exceed the budget, recommend the CHEAPEST available option (economy class, budget hotel).

Plan: {json.dumps(state['plan'], indent=2)}
Output headings: {headings}

╔══════════════════════════════════════════════════════════════╗
║  COORDINATOR'S VALIDATED BRIEF — USE THIS AS PRIMARY SOURCE ║
║  All data below has been cross-validated for consistency.    ║
║  Hotel prices, transport costs, and budget are VERIFIED.     ║
╚══════════════════════════════════════════════════════════════╝

{coordinator_brief}

═══ BUDGET-FILTERED RESEARCH NOTES ═══
(Over-budget options have been removed by the system)

Flight Research:
{chr(10).join('- ' + n for n in flight_notes)}

Hotel Research:
{chr(10).join('- ' + n for n in hotel_notes)}

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
    user_budget = state.get('user_budget') or 0
    num_travelers = state.get('num_travelers', 1)

    structured_reviewer = llm.with_structured_output(ReviewResult)
    review_obj = structured_reviewer.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"""User's original request:
{state['question']}

══════ VALIDATION TARGETS ══════
EXPECTED ITINERARY DAYS: {max(num_days, 1) + 1} — the itinerary must have exactly this many dated days
  (Day 1 = arrival/check-in, middle days = full days, LAST day = check-out/departure).
  The guest stays {max(num_days, 1)} nights. Flag ONLY if the dated-day count differs from {max(num_days, 1) + 1}.
EXPECTED TRAVEL STYLE: {travel_style} — flag any recommendation that doesn't match.
══════════════════════════════════

╔══════════════════════════════════════════════════════════════╗
║  BUDGET VALIDATION — EXACT NUMBERS FROM USER                 ║
╚══════════════════════════════════════════════════════════════╝
USER'S EXACT BUDGET: {int(user_budget):,}
NUMBER OF TRAVELERS: {num_travelers}
MAX TRANSPORT (35%): {int(user_budget * 0.35):,}
MAX HOTEL (45%): {int(user_budget * 0.45):,}

CHECK: Does the draft's total cost exceed {int(user_budget):,}?
If YES → set is_over_budget=True, score MUST be < 50.
CHECK: Does transport exceed {int(user_budget * 0.35):,}?
If YES → flag it in fix_instructions.

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


def _ensure_visa_section(resp: str, visa_notes) -> str:
    """HARD GUARANTEE: the full itinerary must ALWAYS contain a Visa & Entry section.
    If the draft has none, build one from visa_notes (or a safe default) and insert it
    just before the budget section (or at the end). No-op if a visa section already exists."""
    if not resp:
        return resp
    low = resp.lower()
    has_visa = ('🛂' in resp or 'تأشير' in resp or 'التأشيرة' in resp
                or re.search(r'\bvisa\b|\bentry requirements?\b', low))
    if has_visa:
        return resp
    notes = [n.strip() for n in (visa_notes or []) if n and n.strip()]
    body = '\n'.join(f"- {n}" for n in notes) if notes else \
        "- يرجى التحقق من متطلبات التأشيرة وصلاحية جواز السفر قبل السفر / Please verify visa requirements and passport validity before travel."
    section = f"\n\n## 🛂 التأشيرة والدخول\n\n{body}\n"
    m = re.search(r'\n\s*#{0,6}\s*💰', resp)  # insert before the budget block if present
    if m:
        return resp[:m.start()] + section + "\n" + resp[m.start():]
    return resp.rstrip() + section


def _strip_budget_process_sections(resp: str) -> str:
    """Hide Claude's internal budget 'working' from the user: keep only ONE final budget table.
    Removes 'Final Adjustments' / 'To utilize the remaining' narrative sections and any earlier
    DUPLICATE budget tables, keeping the LAST budget table as the single clean one."""
    if not resp:
        return resp
    lines = resp.split('\n')
    def heading_text(l):
        m = re.match(r'^\s*#{1,6}\s+(.*\S)\s*$', l)
        if m:
            return m.group(1)
        m = re.match(r'^\s*\*\*([^*]+)\*\*\s*:?\s*$', l)
        return m.group(1) if m else None
    sections, cur = [], [None, []]
    for l in lines:
        if heading_text(l) is not None:
            sections.append(cur)
            cur = [l, []]
        else:
            cur[1].append(l)
    sections.append(cur)
    drop_re = re.compile(r'final\s+adjustments?|updated\s+budget|to\s+utilize\s+the\s+remaining|'
                         r'التعديلات\s+النهائية|الميزانية\s+المحدّ?ثة|تعديلات\s+الميزانية', re.IGNORECASE)
    budget_re = re.compile(r'budget\s+breakdown|💰|الميزانية', re.IGNORECASE)
    budget_idxs = [i for i, s in enumerate(sections) if s[0] and budget_re.search(s[0])]
    keep_budget = budget_idxs[-1] if budget_idxs else None
    out = []
    for i, s in enumerate(sections):
        head = s[0]
        htext = heading_text(head) if head else None
        if htext and drop_re.search(htext) and i != keep_budget:
            continue
        if head and budget_re.search(head) and keep_budget is not None and i != keep_budget:
            continue
        if i == keep_budget and head:
            s = [re.sub(r'\b(updated|final)\s+', '', head, flags=re.IGNORECASE), s[1]]
        out.append(s)
    res = []
    for head, body in out:
        if head is not None:
            res.append(head)
        res.extend(body)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(res)).strip() + '\n'


def finalizer_node(state: GraphState) -> GraphState:
    num_days = state['plan']['num_days']
    travel_style = state['plan'].get('travel_style', 'mid-range')
    user_budget = state.get('user_budget') or 0
    num_travelers = state.get('num_travelers', 1)
    max_transport = int(user_budget * 0.35) if user_budget else 0
    max_hotel = int(user_budget * 0.45) if user_budget else 0
    # num_days = nights booked (check-out = arrival + num_days).
    # Itinerary spans num_days + 1 dated days (arrival day → check-out day).
    num_nights = max(num_days, 1)
    itinerary_days = num_nights + 1
    max_hotel_per_night = int(max_hotel / num_nights) if max_hotel else 0
    max_transport_per_person = int(max_transport / max(num_travelers, 1)) if max_transport else 0

    # ── user's chosen currency (code + symbol) — used everywhere below ──
    cur_code, cur_sym = (state['user_currency'], CURRENCY_SYMBOLS.get(state['user_currency'], '$')) \
        if state.get('user_currency') else _currency_from_question(state.get('question', ''))
    currency_directive = (
        f"\n╔══════════════════════════════════════════════════════════════╗\n"
        f"║  CURRENCY — MANDATORY: use {cur_code} ({cur_sym}) for EVERY price ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
        f"The traveler chose {cur_code}. EVERY number MUST be in {cur_code} with the symbol '{cur_sym}'. "
        f"The budget {int(user_budget):,} is in {cur_code}.\n"
        f"⚠️ Any 'ج.م' in the format templates is only a PLACEHOLDER — replace it with '{cur_sym}'. "
        f"Do NOT output 'ج.م' unless {cur_code} is EGP. Never mix currencies.\n"
    )

    # ← ORCHESTRATOR PATTERN: Finalizer uses coordinator_brief as PRIMARY source
    coordinator_brief = state.get('coordinator_brief', '')

    # ══════════════════════════════════════════════════════════════
    # If reviewer flagged _force_budget_fix (max iterations reached
    # but still over budget), add EXTRA strict instructions
    # ══════════════════════════════════════════════════════════════
    force_fix = ""
    review_data = state.get("review") or {}
    if review_data.get("_force_budget_fix"):
        budget_result = _extract_total_from_draft(state.get("draft", ""), user_budget)
        force_fix = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ⛔⛔⛔ EMERGENCY BUDGET FIX REQUIRED ⛔⛔⛔                         ║
║  The plan is STILL over budget after {state['iteration']} revision attempts.  ║
║  Current total: ~{int(budget_result['estimated_total']):,} vs budget: {int(user_budget):,}     ║
║  Over by: {int(budget_result['overage']):,}                                     ║
║                                                                      ║
║  FIX IT IN THIS ORDER — do NOT cheapen the whole trip:               ║
║  1) FIRST cut TRANSPORT to the CHEAPEST economy/transit flight        ║
║     (≤ {max_transport:,} total for all travelers). The flight is just      ║
║     transit — this is the biggest saving.                            ║
║  2) KEEP the hotel & activities HIGH. Redirect the money saved on the ║
║     flight INTO the stay: a better hotel, more named restaurants,     ║
║     more excursions/outings. Do NOT downgrade the hotel unless step 1 ║
║     alone still leaves you over budget.                              ║
║  3) Only if STILL over after a cheap flight, trim hotel toward        ║
║     ≤ {max_hotel_per_night:,}/night.                                           ║
║  4) Recalculate ALL totals. Final total MUST be < {int(user_budget):,}         ║
║     and SHOULD use ~85-95% of the budget (don't leave it half-unused).║
╚══════════════════════════════════════════════════════════════════════╝
"""
        print(f"[CODE-LEVEL] Force budget fix in finalizer: over by {int(budget_result['overage']):,}")

    # ══════════════════════════════════════════════════════════════
    # If reviewer flagged _force_budget_upgrade (max iterations
    # reached but budget still underutilized), add upgrade instructions
    # ══════════════════════════════════════════════════════════════
    force_upgrade = ""
    if review_data.get("_force_budget_upgrade") and not force_fix:
        budget_result = _extract_total_from_draft(state.get("draft", ""), user_budget)
        estimated = budget_result.get("estimated_total", 0)
        remaining = user_budget - estimated if estimated else 0
        remaining_pct = (remaining / user_budget * 100) if user_budget else 0
        if remaining_pct > 30:
            target_min = int(user_budget * 0.90)
            target_max = int(user_budget * 0.99)
            fill_amount = int(user_budget * 0.98 - estimated)
            force_upgrade = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ⬆️⬆️⬆️ BUDGET UNDER-USED — FILL IT WITH EXPERIENCES ⬆️⬆️⬆️         ║
║  The plan only uses {int(estimated):,} of {int(user_budget):,} budget ({remaining_pct:.0f}% unused). ║
║  This is a '{travel_style}' trip — the user wants the budget USED.  ║
║                                                                      ║
║  FILL the ~{max(fill_amount,0):,} leftover with REAL, NAMED experiences           ║
║  (do NOT just inflate hotel/flight prices):                          ║
║  1) Premium dining — named restaurants, price/meal, most evenings    ║
║  2) Excursions / day-trips — named, price/person, one per full day   ║
║  3) Outings — spa, private boat, sunset cruise, guided tours         ║
║  4) Optionally upgrade hotel (≤{max_hotel_per_night:,}/night) or flights          ║
║  Add every cost to the budget table. Spread across the daily plan.   ║
║  Target total: {target_min:,} - {target_max:,} (stay UNDER {int(user_budget):,}).        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
            print(f"[CODE-LEVEL] Force budget upgrade in finalizer: only using {int(estimated):,} of {int(user_budget):,}")

    resp = llm.invoke([
        SystemMessage(content=FINALIZER_SYSTEM),
        HumanMessage(content=f"""Question: {state['question']}

══════ CRITICAL CONSTRAINTS ══════
ITINERARY DAYS: {itinerary_days} — write EXACTLY {itinerary_days} dated days. Count them before finishing.
  • Day 1 = ARRIVAL day (check-in ~noon; afternoon/evening only).
  • Days 2 to {num_nights} = FULL days.
  • Day {itinerary_days} = CHECK-OUT / departure day (morning only; check-out + airport transfer).
HOTEL NIGHTS: {num_nights} (checks out on day {itinerary_days}). The check-out day is a separate day — do NOT merge it into the last full day.
TRAVEL STYLE: {travel_style} — every recommendation must match this style.
══════════════════════════════════
{currency_directive}
╔══════════════════════════════════════════════════════════════╗
║  BUDGET HARD LIMIT — TOTAL MUST BE LESS THAN {int(user_budget):,} {cur_sym}       ║
║  Number of travelers: {num_travelers}                                    ║
║  Transport max (35%): {max_transport:,} (ALL travelers combined)         ║
║  Transport max per person: {int(max_transport / max(num_travelers, 1)):,}║
║  Hotel max (45%): {max_hotel:,} (ALL {num_nights} nights combined)       ║
║  Hotel max per night: {max_hotel_per_night:,}                            ║
║  If draft total > {int(user_budget):,}, FIX IT by picking cheaper options.║
╚══════════════════════════════════════════════════════════════╝
{force_fix}{force_upgrade}

Plan: {json.dumps(state['plan'], indent=2)}

╔══════════════════════════════════════════════════════════════╗
║  COORDINATOR'S VALIDATED BRIEF — USE THIS AS PRIMARY SOURCE ║
║  All data below has been cross-validated for consistency.    ║
║  Hotel prices, transport costs, and budget are VERIFIED.     ║
╚══════════════════════════════════════════════════════════════╝

{coordinator_brief}

═══ BUDGET-FILTERED RESEARCH (over-budget options removed) ═══
Flights: {chr(10).join('- ' + n for n in _filter_over_budget_options(state.get('flight_notes', []), int(max_transport / max(num_travelers, 1)) if max_transport else 0, "transport"))}
Hotels: {chr(10).join('- ' + n for n in _filter_over_budget_options(state.get('hotel_notes', []), max_hotel_per_night, "hotel"))}
Visa: {chr(10).join('- ' + n for n in state.get('visa_notes', []))}
Weather: {chr(10).join('- ' + n for n in state.get('weather_notes', []))}
Activities: {chr(10).join('- ' + n for n in state.get('activities_notes', []))}
Places: {chr(10).join('- ' + n for n in state.get('places_notes', []))}
Budget: {chr(10).join('- ' + n for n in state.get('budget_notes', []))}

Current Draft: {state.get('draft', '')}
Review: {json.dumps(state.get('review'), indent=2) if state.get('review') else 'None'}
"""),
    ]).content

    # ══════════════════════════════════════════════════════════════
    # CODE-LEVEL POST-CHECK: If finalizer output is STILL over
    # budget, do a second pass with even stricter instructions
    # ══════════════════════════════════════════════════════════════
    if user_budget:
        final_check = _extract_total_from_draft(resp, user_budget)
        if final_check["is_over_budget"] and final_check["overage"] > user_budget * 0.05:
            print(f"[CODE-LEVEL] Finalizer output STILL over budget by {int(final_check['overage']):,} — forcing second pass")
            resp2 = llm.invoke([
                SystemMessage(content=FINALIZER_SYSTEM),
                HumanMessage(content=f"""The draft below EXCEEDS the budget of {int(user_budget):,}.
Current estimated total: {int(final_check['estimated_total']):,} (over by {int(final_check['overage']):,}).

MANDATORY FIXES — apply ALL of these:
1. Change ALL flights to ECONOMY class. Max transport budget: {max_transport:,} for {num_travelers} travelers.
   That means max {int(max_transport / max(num_travelers, 1)):,} per person for flights.
2. Pick a hotel at max {max_hotel_per_night:,}/night (total hotel max: {max_hotel:,}).
3. Reduce activities/entertainment budget.
4. Recalculate the budget table with CORRECT numbers that sum to LESS than {int(user_budget):,}.

REWRITE the entire draft with these cheaper options. Keep the same format and structure.
Travel style "{travel_style}" means quality experience WITHIN budget, not exceeding it.

Draft to fix:
{resp}
"""),
            ]).content
            # Only use the second pass if it's actually under budget or closer
            check2 = _extract_total_from_draft(resp2, user_budget)
            if not check2["is_over_budget"] or check2["overage"] < final_check["overage"]:
                resp = resp2
                print(f"[CODE-LEVEL] Second pass improved budget: {int(check2['estimated_total']):,} vs {int(final_check['estimated_total']):,}")
            else:
                print(f"[CODE-LEVEL] Second pass didn't improve — keeping first version")

        # ══════════════════════════════════════════════════════════════
        # LAST RESORT — HARD GUARANTEE: if STILL over budget after the LLM
        # passes, the plan must NOT be shipped over budget while cheaper
        # options exist. We (1) try one final LLM pass with the cheapest
        # flight/hotel LOCKED IN, then (2) deterministically rebuild the
        # budget section from the cheapest research options so the result
        # is guaranteed within budget. An over-budget result is only shown
        # when even the cheapest feasible combination exceeds the budget.
        # ══════════════════════════════════════════════════════════════
        final_final = _extract_total_from_draft(resp, user_budget)
        if final_final["is_over_budget"]:
            # Use the user's CHOSEN currency symbol (not whatever the draft happened to print).
            cur = cur_sym

            # cheapest flight (the lever we cut) for the locked LLM pass
            cheap_fp, _ = _cheapest_unit_price(state.get('flight_notes', []), "transport")
            lock_fp = int(cheap_fp) if cheap_fp else int(max_transport_per_person or 0)
            # after a cheap flight, this is how much is left for the stay (hotel + experiences)
            stay_pool = int(user_budget - lock_fp * num_travelers)

            print(f"[CODE-LEVEL] LAST RESORT: still over budget (over by {int(final_final['overage']):,}) — "
                  f"cutting flight to ≤{lock_fp:,}/pp, {stay_pool:,} left for the stay")

            # (1) Final LLM pass: LOCK the cheap flight, then SPEND the rest on a high-level stay.
            if lock_fp:
                resp_locked = llm.invoke([
                    SystemMessage(content=FINALIZER_SYSTEM),
                    HumanMessage(content=f"""The draft below is OVER the budget of {int(user_budget):,}. Rebuild it to FIT —
but do NOT cheapen the whole trip. The concept: the flight is just transit, so cut it to the
cheapest economy option, then put the money saved INTO the stay (a better/higher-level hotel,
more named restaurants, more excursions and outings). This is a '{travel_style}' trip.

RULES:
- Transport: LOCK to the CHEAPEST economy/transit flight ≈ {lock_fp:,}/person → {lock_fp * num_travelers:,} total
  for {num_travelers} travelers. Prefer transit/connecting routes. Do NOT use business class.
- Stay (hotel + activities + dining): spend about {stay_pool:,} — RAISE the hotel level and add named
  premium restaurants, excursions, and outings across the days. Keep it luxurious, just within budget.
- The FINAL TOTAL MUST be < {int(user_budget):,}, and SHOULD use ~85-95% of the budget (don't leave it half-unused).
- Recalculate the budget table. Keep the same language, format, structure, and number of itinerary days.

Draft to fix:
{resp}
"""),
                ]).content
                check_locked = _extract_total_from_draft(resp_locked, user_budget)
                if not check_locked["is_over_budget"]:
                    resp = resp_locked
                    print(f"[CODE-LEVEL] Locked pass fit the budget: {int(check_locked['estimated_total'] or 0):,}")

            # (2) DETERMINISTIC GUARANTEE — rebuild the budget section no matter what the LLM did.
            resp, fit_status = _force_budget_fit(
                resp, state.get('flight_notes', []), state.get('hotel_notes', []),
                user_budget, num_nights, num_travelers,
                max_transport_per_person, max_hotel_per_night, cur,
                travel_style=travel_style,
            )
            print(f"[CODE-LEVEL] _force_budget_fit status: {fit_status}")

        # ══════════════════════════════════════════════════════════════
        # POST-CHECK: If budget is underutilized for a non-budget traveler,
        # FILL the leftover with real named experiences (dining/excursions/
        # outings) instead of leaving money unused. Target ~90% utilization.
        # Triggers even for small leftovers (e.g. 3-4% of a large budget) on
        # trips of 3+ nights — one more nice dinner/excursion is worth adding.
        # ══════════════════════════════════════════════════════════════
        final_util = _extract_total_from_draft(resp, user_budget) or {}
        est_val = final_util.get("estimated_total") or 0
        if not final_util.get("is_over_budget") and est_val > 0:
            est = est_val
            rem = user_budget - est
            rem_pct = (rem / user_budget) * 100
            style_lower = travel_style.lower()
            is_budget_style = style_lower in ['budget', 'اقتصادي', 'رخيص']
            # Fill when: non-budget style, trip has 3+ nights, and either a meaningful
            # % is unused (>7%) OR a meaningful absolute amount is unused (enough for
            # at least one extra experience — ~2% of budget or 2,500, whichever larger).
            min_fill_abs = max(int(user_budget * 0.02), 2500)
            should_fill = (
                not is_budget_style
                and num_nights >= 3
                and (rem_pct > 7 or rem >= min_fill_abs)
                and rem >= min_fill_abs
            )
            if should_fill:
                print(f"[CODE-LEVEL] Finalizer leftover {int(rem):,} ({rem_pct:.0f}%) — filling with experiences toward ~99%")
                target_min = int(user_budget * 0.95)
                target_max = int(user_budget * 0.99)
                fill_amount = int(user_budget * 0.98 - est)
                resp_upgrade = llm.invoke([
                    SystemMessage(content=FINALIZER_SYSTEM),
                    HumanMessage(content=f"""The draft below uses only {int(est):,} of the {int(user_budget):,} budget — {int(rem):,} ({rem_pct:.0f}%) is left unused.
This is a '{travel_style}' trip of {num_nights} nights. The user wants the budget USED on a great trip, not left over.

FILL the ~{max(fill_amount, 0):,} leftover with REAL, NAMED experiences — do NOT inflate hotel/flight prices.
⛔ Do NOT write a note like "الميزانية المتبقية: X، يمكن استخدامها لـ..." — ADD the experiences yourself instead.
✅ Repeating a standout place is FINE (a 2nd dinner at the same great restaurant, a 2nd excursion) to use the budget.
1. Premium dining — add named restaurants (with price/meal). Aim for one nice dinner on most evenings.
2. Excursions / day-trips — named operator + location + price/person. Add roughly one per full day.
3. Outings & add-ons — spa, private boat, sunset cruise, guided tours, premium entry tickets.
4. Only if it genuinely improves the trip, you may also upgrade the hotel (≤{max_hotel_per_night:,}/night)
   or flights (≤{int(max_transport / max(num_travelers, 1)):,}/person).

Spread the additions across the day-by-day itinerary (keep the SAME number of dated days) and add
every new cost to the budget table under Activities / Dining. Recalculate the totals.

Target final total: {target_min:,} - {target_max:,}. HARD RULE: stay UNDER {int(user_budget):,} — never exceed it.
Keep the same format, structure, and number of itinerary days.

Draft to enrich:
{resp}
"""),
                ]).content
                # Only use the fill pass if it uses more budget but is still within budget.
                check_upgrade = _extract_total_from_draft(resp_upgrade, user_budget)
                upgrade_est = check_upgrade.get("estimated_total", 0)
                if upgrade_est and not check_upgrade["is_over_budget"] and upgrade_est > est:
                    resp = resp_upgrade
                    print(f"[CODE-LEVEL] Fill pass improved utilization: {int(upgrade_est):,} vs {int(est):,}")
                elif upgrade_est and check_upgrade["is_over_budget"]:
                    print(f"[CODE-LEVEL] Fill pass went over budget ({int(upgrade_est):,}) — keeping original")
                else:
                    print(f"[CODE-LEVEL] Fill pass didn't improve — keeping original")

    # ══════════════════════════════════════════════════════════════
    # FOOD CONSISTENCY — the budget table must agree with the itinerary.
    # If the plan lists paid meals but the Food row is 0/missing (or far below the
    # real meal total), the budget is self-contradictory. Fix it with ONE pass that
    # KEEPS the restaurants luxurious but REDUCES how many premium meals are scheduled
    # so the (now-counted) food fits, the total stays within budget, and ~90-97% is used.
    # ══════════════════════════════════════════════════════════════
    if user_budget:
        style_low = travel_style.lower()
        is_budget_style = style_low in ['budget', 'اقتصادي', 'رخيص']
        food_in_itin = _sum_itinerary_food(resp, num_travelers)
        food_in_table = _extract_food_line(resp)
        cur_check = _extract_total_from_draft(resp, user_budget)
        cur_total = cur_check.get("estimated_total") or 0
        remaining_pct = ((user_budget - cur_total) / user_budget * 100) if cur_total else 0

        meals_present = food_in_itin >= max(50, user_budget * 0.03)
        food_missing = (food_in_table is None) or (food_in_table < food_in_itin * 0.5)
        underused = (not is_budget_style) and remaining_pct > 8

        if (meals_present and food_missing) or underused:
            # How much the stay can really spend on food: budget minus transport+hotel, keep ~1% buffer,
            # leave room for the existing activities. Target ~98-99% utilization overall.
            spendable = int(user_budget * 0.99)
            food_target = int(min(food_in_itin if food_in_itin > 0 else user_budget * 0.18,
                                  max(user_budget * 0.12, spendable - cur_total + (food_in_table or 0))))
            food_target = max(food_target, int(user_budget * 0.10))
            reason = ("Food row is 0/too low while the itinerary lists paid restaurants"
                      if (meals_present and food_missing)
                      else f"{remaining_pct:.0f}% of the budget is unused")
            print(f"[CODE-LEVEL] Food-consistency pass: {reason}. "
                  f"itinerary_food≈{int(food_in_itin):,}, table_food={food_in_table}, target≈{food_target:,}")
            resp_food = llm.invoke([
                SystemMessage(content=FINALIZER_SYSTEM),
                HumanMessage(content=f"""Fix the budget so it is CONSISTENT and well-used. Do NOT cheapen the trip.

PROBLEM: {reason}.
The budget table must AGREE with the day-by-day itinerary.

RULES:
1. The 'الطعام / Food' row MUST equal the sum of the paid meals in the itinerary
   (per-person meals × {num_travelers} travelers, group meals as-is). Food = 0 is FORBIDDEN while any
   paid restaurant is scheduled.
2. KEEP the restaurants LUXURIOUS (Michelin / premium are great) — but REDUCE HOW MANY expensive
   dinners are scheduled: feature 2–3 signature premium dinners across the whole trip, and make the
   other evenings lighter (a moderate local restaurant, or a meal included in the hotel). Quality over
   quantity. Food spend should land around {food_target:,} {cur_sym}.
3. Recompute every row (Transport, Hotel, Activities, FOOD, Misc) and the TOTAL. The TOTAL MUST include
   food and be LESS than {int(user_budget):,} {cur_sym}, while USING about 97-100% of it (only a tiny remaining).
4. Keep the SAME language, format, currency ({cur_code} / {cur_sym}), and number of itinerary days.

Draft to fix:
{resp}
"""),
            ]).content
            chk = _extract_total_from_draft(resp_food, user_budget)
            new_food = _extract_food_line(resp_food)
            new_total = chk.get("estimated_total") or 0
            # Adopt only if it's within budget AND food is now counted (non-trivial) AND it didn't shrink the total.
            if (not chk["is_over_budget"]) and new_food and new_food > 0 and new_total >= cur_total * 0.9:
                resp = resp_food
                print(f"[CODE-LEVEL] Food-consistency pass applied: food={int(new_food):,}, total={int(new_total):,}")
            else:
                print(f"[CODE-LEVEL] Food-consistency pass rejected (food={new_food}, total={int(new_total):,}, "
                      f"over={chk['is_over_budget']}) — keeping original")

    # ══════════════════════════════════════════════════════════════
    # REAL RECOMMENDED-HOTEL PRICE — pull the ACTUAL nightly price of the chosen
    # hotel from live search (free Tavily) and use it, so the headline hotel price
    # matches reality instead of the LLM's guess. Band-clamped as a safety net; on
    # any search miss it silently keeps the band value. Runs BEFORE the clamp &
    # budget so every downstream number uses the real figure.
    # ══════════════════════════════════════════════════════════════
    _dest_h = (state.get('plan') or {}).get('destination', '')
    try:
        resp_real, real_changed = _apply_real_recommended_price(resp, _dest_h, cur_sym)
        if real_changed:
            resp = resp_real
            print("[CODE-LEVEL] Recommended hotel price set from live search")
    except Exception as _real_err:
        print(f"[CODE-LEVEL] Real hotel price lookup error (ignored): {_real_err}")

    # ══════════════════════════════════════════════════════════════
    # HOTEL PRICE SANITY CLAMP — pull every hallucinated hotel/night price into a
    # realistic band for the destination & star tier, BEFORE the budget is rebuilt
    # (so the budget uses the corrected price). Stops the '2,079 for a 5-star in
    # Dahab' / '4,000 for a cheap hotel' hallucinations deterministically.
    # ══════════════════════════════════════════════════════════════
    try:
        resp_clamped, clamp_changed = _clamp_hotel_prices_in_draft(resp, _dest_h, cur_sym)
        if clamp_changed:
            resp = resp_clamped
            print("[CODE-LEVEL] Hotel prices clamped to realistic bands")
    except Exception as _clamp_err:
        print(f"[CODE-LEVEL] Hotel price clamp error (ignored): {_clamp_err}")

    # ══════════════════════════════════════════════════════════════
    # DETERMINISTIC TRANSPORT + BUDGET — single source of truth for the transport
    # choice/price AND all budget numbers. Transport is chosen in code (fly to the
    # nearest airport + transfer for no-airport destinations, round trip, realistic
    # prices) so the LLM can't hallucinate it; the budget is then rebuilt in code
    # from that transport + the recommended hotel × nights. Transport is committed
    # ONLY when the whole budget fits, so the transport table and the budget can
    # never disagree. On any doubt, the draft is left untouched.
    # ══════════════════════════════════════════════════════════════
    try:
        if user_budget:
            _dest = (state.get('plan') or {}).get('destination', '')
            t_info = _compute_transport(state.get('question', ''), _dest,
                                        num_travelers, travel_style, cur_sym)
            # LIVE FLIGHT-PRICE VERIFICATION — refine the route table's per-person fare with the
            # real round-trip price from live search. It can only RAISE the fare toward reality
            # (a 'from $X' hit is often one-way/promo, so never lower it below the realistic
            # table floor). Keeps per-person × travelers exact. Badge shown in the section.
            if t_info:
                try:
                    _live_pp = _real_flight_roundtrip_price(
                        _trip_origin(state.get('question', '')), _dest, cur_sym)
                    _anchor = t_info.get("per_person") or 0
                    if _live_pp and _anchor:
                        _refined = min(max(_live_pp, _anchor * 0.85), _anchor * 2.2)
                        if _refined > _anchor * 1.08:          # reality is meaningfully higher
                            _trav = t_info.get("trav") or max(int(num_travelers or 1), 1)
                            t_info = dict(t_info)
                            t_info["per_person"] = int(round(_refined))
                            t_info["total"] = int(round(_refined * _trav))
                            t_info["_price_verified"] = True
                            print(f"[CODE-LEVEL] Flight price verified from live search: "
                                  f"{t_info['per_person']:,}/person (was {int(_anchor):,})")
                except Exception as _fp_err:
                    print(f"[CODE-LEVEL] Live flight price error (ignored): {_fp_err}")
            override = t_info["total"] if t_info else None
            resp_bud, bud_applied, bud_reason = _rebuild_budget_deterministic(
                resp, user_budget, num_nights, num_travelers, cur_sym, travel_style,
                state.get("hotel_notes"), state.get("flight_notes"),
                transport_override=override)
            if bud_applied:
                resp = resp_bud
                print(f"[CODE-LEVEL] Deterministic budget rebuilt: {bud_reason}")
            else:
                print(f"[CODE-LEVEL] Deterministic budget NOT rebuilt ({bud_reason}) — keeping draft budget")
            # Transport table is committed WHENEVER we have a known route — independent of the
            # budget rebuild — because the deterministic section is self-consistent by
            # construction (per-person × travelers = total). This is what stops the transport
            # table from ever showing inconsistent numbers when the budget rebuild is skipped.
            if t_info:
                resp = _replace_section(
                    resp, ["وسائل النقل", "النقل", "المواصلات", "Transport", "Flights"],
                    _build_transport_section(t_info, cur_sym))
                print(f"[CODE-LEVEL] Deterministic transport applied: {t_info['mode']} "
                      f"(total={t_info['total']:,})")
    except Exception as _tb_err:
        print(f"[CODE-LEVEL] Deterministic transport/budget error (ignored): {_tb_err}")

    # ══════════════════════════════════════════════════════════════
    # TRANSPORT MATH GUARANTEE — for ANY destination (including those the curated
    # route tables don't cover), force Total == per-person × travelers in the
    # transport table. Rows already written as 'pp × n = total' are left alone.
    # ══════════════════════════════════════════════════════════════
    try:
        resp_tm, tm_changed = _enforce_transport_math(resp, num_travelers, cur_sym)
        if tm_changed:
            resp = resp_tm
            print("[CODE-LEVEL] Transport total corrected to per-person × travelers")
    except Exception as _tm_err:
        print(f"[CODE-LEVEL] Transport math pass error (ignored): {_tm_err}")

    # ══════════════════════════════════════════════════════════════
    # HOTEL ↔ BUDGET CONSISTENCY — the budget's hotel line MUST reflect the
    # RECOMMENDED hotel's price × nights. We only act when the budget clearly
    # used a CHEAPER hotel than recommended (amount well BELOW recommended×nights);
    # a HIGHER figure is left alone (it may legitimately cover several rooms).
    # Detection is deterministic; the fix is a single targeted LLM re-ask that
    # keeps the draft's own table format. Fully guarded — on any doubt, do nothing.
    # ══════════════════════════════════════════════════════════════
    try:
        if user_budget:
            rec_night = _recommended_hotel_night_price(resp, state.get("hotel_notes"), travel_style)
            cur_hotel = _budget_hotel_line_value(resp)
            nights_n = max(int(num_nights or 1), 1)
            if rec_night and cur_hotel:
                expected_hotel = rec_night * nights_n
                if cur_hotel < expected_hotel * 0.88:   # budget quietly used a cheaper hotel
                    print(f"[CODE-LEVEL] Hotel/budget mismatch: budget hotel={int(cur_hotel):,} but "
                          f"recommended {int(rec_night):,}/night × {nights_n} nights = {int(expected_hotel):,}. Re-asking.")
                    resp_hotel = llm.invoke([
                        SystemMessage(content=FINALIZER_SYSTEM),
                        HumanMessage(content=f"""The budget table in the draft is INCONSISTENT with the recommended hotel.

The hotel marked 🏆 توصية الـAI costs {int(rec_night):,} {cur_sym} per night and the trip is {nights_n} nights,
so the 'الإقامة / الفندق' line in the budget MUST be {int(rec_night):,} × {nights_n} = {int(expected_hotel):,} {cur_sym}
(multiply further by the number of rooms if the group needs more than one room).
Right now the budget shows only {int(cur_hotel):,} {cur_sym} — it quietly used a CHEAPER hotel than the recommended one. Fix it.

RULES:
1. Set the accommodation/hotel line to the RECOMMENDED hotel's real cost ({int(expected_hotel):,} {cur_sym} for {nights_n} nights, × rooms if needed).
2. Recompute the TOTAL and remaining budget. The TOTAL must stay UNDER {int(user_budget):,} {cur_sym}.
3. If raising the hotel pushes the total over budget, reduce OTHER lines (fewer premium dinners / activities) — do NOT switch back to a cheaper hotel than the recommended one.
4. Keep the SAME language, table format, currency ({cur_code} / {cur_sym}), hotels table, and number of itinerary days.

Draft to fix:
{resp}
""")]).content
                    chk_h = _extract_total_from_draft(resp_hotel, user_budget)
                    new_hotel = _budget_hotel_line_value(resp_hotel)
                    # Adopt only if the hotel line now matches (±8%) AND the plan stays within budget.
                    if (not chk_h["is_over_budget"]) and new_hotel and new_hotel >= expected_hotel * 0.92:
                        resp = resp_hotel
                        print(f"[CODE-LEVEL] Hotel-consistency pass applied: hotel now {int(new_hotel):,}")
                    else:
                        print(f"[CODE-LEVEL] Hotel-consistency pass rejected (hotel={new_hotel}, "
                              f"over={chk_h['is_over_budget']}) — keeping original")
    except Exception as _hc_err:
        print(f"[CODE-LEVEL] Hotel-consistency pass error (ignored): {_hc_err}")

    # ══════════════════════════════════════════════════════════════
    # STRUCTURE GUARANTEES (additive, run after all budget work):
    #  • Hide the internal budget 'working' — keep ONE clean final budget table.
    #  • The Visa & Entry section must ALWAYS be present (even for domestic trips).
    # ══════════════════════════════════════════════════════════════
    try:
        resp = _strip_budget_process_sections(resp)
        _before_visa = resp
        resp = _ensure_visa_section(resp, state.get('visa_notes', []))
        if resp != _before_visa:
            print("[CODE-LEVEL] Visa section was missing — injected from visa_notes")
    except Exception as _struct_err:
        print(f"[CODE-LEVEL] Structure-guarantee pass error (ignored): {_struct_err}")

    # NOTE: the hotel price verification is already shown ONCE, as the badge under the
    # recommended hotel (see _apply_real_recommended_price). We deliberately do NOT add a
    # second reviewer note near the budget — that was redundant clutter. The price check still
    # runs in the review loop (back-end); it just isn't printed twice on the page.

    # ══════════════════════════════════════════════════════════════
    # CURRENCY NORMALIZATION — HARD GUARANTEE: force the whole output into the
    # user's chosen currency symbol. Fixes the "USD trip printed in ج.م" bug
    # regardless of what the LLM/templates produced. Numbers stay; symbol is fixed.
    # ══════════════════════════════════════════════════════════════
    resp = _normalize_currency_in_text(resp, cur_sym)
    for key in ('flight_notes', 'hotel_notes', 'activities_notes', 'places_notes'):
        notes = state.get(key)
        if notes:
            state[key] = [_normalize_currency_in_text(n, cur_sym) for n in notes]
    print(f"[CODE-LEVEL] Currency normalized to {cur_code} ({cur_sym})")

    state["draft"] = resp

    # ══════════════════════════════════════════════════════════════
    # SYNC budget_notes FROM the final draft — so tab4 matches tab1.
    # Use the USER'S chosen currency symbol, never a guessed default.
    # ══════════════════════════════════════════════════════════════
    currency = cur_sym

    synced_notes = _generate_budget_notes_from_draft(resp, user_budget, currency)
    if synced_notes:
        state["budget_notes"] = synced_notes
        print(f"[CODE-LEVEL] Budget notes synced from draft: {synced_notes}")

    return state


# ─────────────────────────────────────────────
# 7. Conditional Edge
# ─────────────────────────────────────────────

def should_revise(state: GraphState) -> Literal["revise", "finalize"]:
    review = state["review"]
    score = review["score"]

    # ══════════════════════════════════════════════════════════════
    # CODE-LEVEL BUDGET CHECK — runs BEFORE max_iterations check
    # so an over-budget plan NEVER passes through unchecked
    # ══════════════════════════════════════════════════════════════
    user_budget = state.get("user_budget")
    is_over_budget = False
    if user_budget and state.get("draft"):
        budget_result = _extract_total_from_draft(state["draft"], user_budget)
        if budget_result["is_over_budget"]:
            is_over_budget = True
            overage = budget_result["overage"]
            estimated = budget_result["estimated_total"]
            print(f"[CODE-LEVEL] OVER BUDGET DETECTED: total={estimated}, budget={user_budget}, overage={overage}")
            # ── Force the review to reflect over-budget ──
            if "budget_check" not in review:
                review["budget_check"] = {}
            review["budget_check"]["is_over_budget"] = True
            review["budget_check"]["overage_amount"] = overage
            review["budget_check"]["estimated_total"] = estimated
            review["budget_check"]["user_budget"] = user_budget
            # ── Force score below 80 ──
            review["score"] = min(review["score"], 50)
            score = review["score"]
            # ── Add specific fix instructions ──
            if "fix_instructions" not in review:
                review["fix_instructions"] = []
            review["fix_instructions"].insert(0,
                f"CRITICAL: Plan is OVER BUDGET by {int(overage):,}. "
                f"Total={int(estimated):,} but budget={int(user_budget):,}. "
                f"FIX IT IN THIS ORDER (do NOT cheapen the whole trip): "
                f"1) FIRST cut TRANSPORT — switch to the CHEAPEST economy/transit flight. The flight is "
                f"just transit; this is the biggest saving and frees money for the rest. "
                f"2) KEEP the hotel and the activities/dining at a HIGH level — redirect the money saved on "
                f"the flight INTO the stay (better hotel, more named restaurants, more excursions/outings). "
                f"Do NOT downgrade the hotel unless step 1 alone still leaves you over budget. "
                f"3) Only if still over after minimizing the flight, trim the hotel toward "
                f"{int(user_budget * 0.45 / max(state['plan']['num_days'], 1)):,}/night. "
                f"The final total MUST be LESS than {int(user_budget):,}, and should USE ~85-95% of it."
            )
            state["review"] = review

    # ══════════════════════════════════════════════════════════════
    # CODE-LEVEL UNDER-UTILIZATION CHECK — for luxury/mid-range
    # If >40% budget remaining, the plan picked too-cheap options
    # ══════════════════════════════════════════════════════════════
    is_underutilized = False
    if user_budget and state.get("draft") and not is_over_budget:
        budget_result = _extract_total_from_draft(state["draft"], user_budget)
        estimated = budget_result.get("estimated_total", 0)
        if estimated and estimated > 0:
            remaining = user_budget - estimated
            remaining_pct = (remaining / user_budget) * 100
            travel_style = state['plan'].get('travel_style', 'mid-range').lower()
            num_days = state['plan'].get('num_days', 1)
            num_nights = max(num_days, 1)  # num_days = nights booked
            num_travelers = state.get('num_travelers', 1)

            # Thresholds: luxury >35% remaining, mid-range >45% remaining
            threshold = 35 if 'lux' in travel_style or 'فاخر' in travel_style else 45
            if remaining_pct > threshold and travel_style not in ['budget', 'اقتصادي', 'رخيص']:
                is_underutilized = True
                print(f"[CODE-LEVEL] BUDGET UNDERUTILIZED: total={int(estimated):,}, "
                      f"budget={int(user_budget):,}, remaining={int(remaining):,} ({remaining_pct:.0f}%)")

                if "budget_check" not in review:
                    review["budget_check"] = {}
                review["budget_check"]["budget_underutilized"] = True
                review["budget_check"]["remaining_amount"] = remaining
                review["budget_check"]["remaining_percentage"] = remaining_pct

                review["score"] = min(review["score"], 65)
                score = review["score"]

                if "fix_instructions" not in review:
                    review["fix_instructions"] = []

                # Calculate what they SHOULD spend
                target_hotel_per_night = int(user_budget * 0.45 / num_nights)
                target_transport_per_person = int(user_budget * 0.35 / max(num_travelers, 1))

                remaining_to_fill = int(user_budget * 0.98 - estimated)
                review["fix_instructions"].insert(0,
                    f"BUDGET UNDERUTILIZED: Only {int(estimated):,} of {int(user_budget):,} used "
                    f"({remaining_pct:.0f}% remaining ≈ {int(remaining):,} unused). This is a '{travel_style}' trip — "
                    f"the user wants the budget USED on a great trip, not left over. "
                    f"FILL the remaining ~{max(remaining_to_fill, 0):,} with REAL, NAMED experiences "
                    f"(do NOT just inflate hotel/flight prices). "
                    f"⛔ Do NOT write 'الميزانية المتبقية: X، يمكن استخدامها لـ...' — ADD the experiences yourself. "
                    f"✅ Repeating a standout place is fine (2nd dinner at the same restaurant, a 2nd excursion): "
                    f"1) Add premium dining — named restaurants with price/meal, one nice dinner most evenings. "
                    f"2) Add excursions/day-trips — named operator + location + price/person, one per full day. "
                    f"3) Add outings/add-ons — spa, private boat, sunset cruise, guided tours, premium tickets. "
                    f"4) Optionally upgrade the hotel (up to {target_hotel_per_night:,}/night) or flights "
                    f"(up to {target_transport_per_person:,}/person) if that genuinely improves the trip. "
                    f"Spread everything across the daily itinerary and add the costs to the budget table. "
                    f"Target total: {int(user_budget * 0.95):,}-{int(user_budget * 0.99):,} (stay UNDER {int(user_budget):,})."
                )
                state["review"] = review
    # ══════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════
    # REAL HOTEL-PRICE VERIFICATION (loop-level) — the reviewer live-searches the
    # recommended hotel's price and compares it to what the draft shows. On a
    # mismatch (>22%) it writes a note, lowers the score, injects the EXACT real
    # price as a fix instruction, and sends the plan back to revise. Cached, so the
    # search runs once per hotel. The finalizer stays as the deterministic backstop.
    # ══════════════════════════════════════════════════════════════
    price_mismatch = False
    if state.get("draft"):
        try:
            _dest = state['plan'].get('destination', '')
            _cc, _cs = (state['user_currency'], CURRENCY_SYMBOLS.get(state['user_currency'], '$')) \
                if state.get('user_currency') else _currency_from_question(state.get('question', ''))
            _ts = state['plan'].get('travel_style', 'mid-range')
            pv = _verify_recommended_hotel_price(state["draft"], _dest, _cs, _ts)
            review["price_check"] = pv
            if pv.get("mismatch") and pv.get("real") and pv.get("displayed"):
                price_mismatch = True
                review["score"] = min(review.get("score", 100), 60)
                review.setdefault("fix_instructions", []).insert(0,
                    f"HOTEL PRICE WRONG vs reality: the recommended hotel '{pv.get('name')}' shows "
                    f"{int(pv['displayed']):,} {_cs}/night, but live web search says the REAL price is "
                    f"~{int(pv['real']):,} {_cs}/night. Set that hotel's price/night to EXACTLY "
                    f"{int(pv['real']):,} {_cs} in the hotel table and update the budget hotel line to match. "
                    f"Do NOT invent a different number.")
                print(f"[CODE-LEVEL] REVIEWER price check — MISMATCH: '{pv.get('name')}' "
                      f"displayed={int(pv['displayed']):,}, real={int(pv['real']):,} {_cs} → revise")
            else:
                print(f"[CODE-LEVEL] REVIEWER price check: {pv.get('note', '')}")
            state["review"] = review
            score = review["score"]
        except Exception as _pv_err:
            print(f"[CODE-LEVEL] Price verification error (ignored): {_pv_err}")

    # Max iterations — but if STILL over budget on last iteration,
    # we flag it so finalizer can handle it
    if state["iteration"] >= state["max_iterations"]:
        if is_over_budget:
            print(f"[CODE-LEVEL] Max iterations reached but STILL over budget — flagging for finalizer")
            state["review"]["_force_budget_fix"] = True
        if is_underutilized:
            print(f"[CODE-LEVEL] Max iterations reached but STILL underutilized — flagging for finalizer")
            state["review"]["_force_budget_upgrade"] = True
        return "finalize"

    # If over budget, force revision (we have iterations left)
    if is_over_budget:
        return "revise"

    # If underutilized, force revision (we have iterations left)
    if is_underutilized:
        return "revise"

    # If the recommended hotel's price disagrees with live search, revise with the real figure
    if price_mismatch:
        return "revise"

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
        "max_iterations": 3,
        "user_budget": 3000.0,
        "num_travelers": 1,
    }

    print("\n⏳ Running multi-agent travel planner...\n")
    result = app.invoke(initial_state)

    print("=" * 60)
    print("✈️  FINAL TRAVEL ITINERARY")
    print("=" * 60)
    print(result["draft"])

    if result.get("review"):
        print(f"\n📊 Review score: {result['review']['score']}/100")
