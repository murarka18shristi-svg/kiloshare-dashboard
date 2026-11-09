"""The KiloShare consumer survey: 25 questions, each mapped to the algorithm(s) it feeds."""

LIKERT = ['1 Strongly disagree', '2 Disagree', '3 Neutral', '4 Agree', '5 Strongly agree']

FEATURES = {
    'F1_same_flight_matching': 'Matching with a passenger on the same flight',
    'F2_id_verified_profiles': 'ID-verified profiles for every user',
    'F3_in_app_chat': 'In-app chat before meeting',
    'F4_counter_inspection_sealed_pouch': 'Counter inspection and a sealed, tamper-evident pouch',
    'F5_transit_insurance': 'Insurance on the items you send',
    'F6_live_item_tracking': 'Live tracking of your pouch',
    'F7_ratings_reviews': 'Ratings and reviews of other users',
    'F8_courier_forwarding': 'Courier forwarding when no match is found',
    'F9_repack_weigh_station': 'Repack and weigh station at the airport',
    'F10_wallet_credits': "Wallet credits for carrying someone's pouch",
}

CONCEPT = ("Over your 15 kg allowance? Airlines charge up to Rs 800 for every extra kg at the airport. "
           "KiloShare matches you with a passenger on the same flight who has spare allowance. You both meet at a staffed "
           "KiloShare counter, where staff check your items, seal them in a tamper-evident pouch and record both IDs. "
           "Your co-passenger checks the pouch in with their bag and you collect it at the KiloShare counter on arrival. "
           "You pay much less than the airline's excess fee and your co-passenger earns credits. No electronics, cash, "
           "jewellery, medicines, liquids or sealed packages. If no one on your flight has spare space, KiloShare can "
           "courier your extra items to your destination instead (arrives in 2-3 days).")

C, R, A, G, P = 'Classification', 'Regression', 'Association rules', 'Clustering', 'Profiling'
QUESTIONS = [
    # id, section, question, answer type / options, dataset column, algorithm roles
    ('Q1', 'Screener', 'Have you taken at least one domestic flight within India in the last 12 months?', 'Yes / No (No ends the survey)', '(not stored)', []),
    ('Q2', 'A. About you', 'What is your age?', 'Number, 18+', 'age', [f'{C}: predictor', f'{G}: input']),
    ('Q3', 'A. About you', 'Which city do you live in?', 'Mumbai / Delhi / Bengaluru / Hyderabad / Chennai / Kolkata / Pune / Ahmedabad', 'city', [f'{P}: segment and city view']),
    ('Q4', 'A. About you', 'What is your occupation?', 'Student / Salaried / Business owner / Freelancer / Homemaker / Retired', 'occupation', [f'{P}: segment personas']),
    ('Q5', 'A. About you', 'What is your monthly household income?', '<Rs 25k / 25k-50k / 50k-1L / 1L-2L / 2L+', 'monthly_income_band', [f'{C}: predictor', f'{R}: driver of price']),
    ('Q6', 'B. Travel and baggage', 'How many domestic flights did you take in the last 12 months? (each one-way = 1)', 'Number', 'domestic_flights_12m', [f'{C}: predictor', f'{G}: input']),
    ('Q7', 'B. Travel and baggage', 'Which fare type do you usually book?', 'Saver/basic / Flexi/value / Business/premium', 'usual_fare_type', [f'{R}: driver of price']),
    ('Q8', 'B. Travel and baggage', 'In the last 12 months, how many times was your checked bag over the free allowance?', 'Number (0 if never)', 'times_exceeded_allowance_12m', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q9', 'B. Travel and baggage', 'On those trips, by roughly how many kg were you over, on average?', 'Kg (0 if never)', 'avg_excess_kg', [f'{C}: predictor', f'{R}: driver', 'Break-even: kg per transfer']),
    ('Q10', 'B. Travel and baggage', 'What does your checked bag usually weigh?', 'Kg', 'avg_checked_bag_kg', [f'{C}: predictor', f'{G}: input', 'Break-even: carriers with spare room']),
    ('Q11', 'B. Travel and baggage', 'Roughly how much did you pay in excess-baggage fees in the last 12 months?', 'Rs (0 if none)', 'excess_fees_paid_12m_inr', [f'{C}: predictor', f'{R}: driver']),
    ('Q12', 'C. Attitudes (1-5)', 'I do not like other people handling my luggage.', 'Likert 1-5', 'possessiveness_luggage', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q13', 'C. Attitudes (1-5)', 'I could trust a verified stranger on my flight.', 'Likert 1-5', 'trust_strangers', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q14', 'C. Attitudes (1-5)', "I worry about security problems from carrying someone else's items.", 'Likert 1-5', 'security_concern', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q15', 'C. Attitudes (1-5)', 'Saving money on baggage fees matters a lot to me.', 'Likert 1-5', 'price_sensitivity', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q16', 'C. Attitudes (1-5)', 'I am comfortable using new travel apps.', 'Likert 1-5', 'tech_comfort', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q17', 'C. Attitudes (1-5)', 'A staffed service counter at the airport would make me trust a new travel service.', 'Likert 1-5', 'counter_trust_boost', [f'{C}: predictor', f'{G}: input', f'{R}: driver']),
    ('Q18', 'D. Reaction to KiloShare (after the concept card)', 'Given the opportunity, would you use KiloShare to send your extra kilos?', 'Yes / No', 'would_use_app', [f'{C}: TARGET (demand)']),
    ('Q19', 'D. Reaction to KiloShare (after the concept card)', 'How likely are you to use KiloShare in the next 6 months?', '1 Not at all likely ... 5 Definitely', 'usage_intent_6m', [f'{G}: high / medium / low conversion bands', 'Break-even: intent calibration']),
    ('Q20', 'D. Reaction to KiloShare (after the concept card)', 'Which of these features would you want in KiloShare? Select all that apply.', 'Multi-select, 10 features', 'F1 ... F10 (0/1)', [f'{A}: one basket per respondent']),
    ('Q21', 'D. Reaction to KiloShare (after the concept card)', 'What is the most you would pay per extra kg to send it with a co-passenger on your flight?', 'Rs per kg', 'wtp_p2p_per_kg_inr', [f'{R}: TARGET (peer-to-peer price)']),
    ('Q22', 'D. Reaction to KiloShare (after the concept card)', 'What is the most you would pay per extra kg to have it couriered to your destination in 2-3 days?', 'Rs per kg', 'wtp_forwarding_per_kg_inr', [f'{R}: TARGET (forwarding price)']),
    ('Q23', 'D. Reaction to KiloShare (after the concept card)', 'If you had spare allowance, would you carry a verified, sealed pouch for a co-passenger?', 'Yes / No', 'would_carry', [f'{C}: TARGET (supply)']),
    ('Q24', 'D. Reaction to KiloShare (after the concept card)', 'What is the minimum credit per kg you would need to carry it?', 'Rs per kg', 'min_credit_per_kg_inr', [f'{R}: TARGET (carrier price)']),
    ('Q25', 'D. Reaction to KiloShare (after the concept card)', 'What is your biggest concern about a service like this?', 'Open text', 'biggest_concern_text', ['Text coding: concern themes']),
]

LIKERT_COLS = ['possessiveness_luggage', 'trust_strangers', 'security_concern', 'price_sensitivity', 'tech_comfort', 'counter_trust_boost']
FEATURE_COLS = list(FEATURES)
