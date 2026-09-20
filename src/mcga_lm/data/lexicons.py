

from __future__ import annotations

from typing import Tuple

FIRST_NAMES: Tuple[str, ...] = (
    "Aisha", "Benjamin", "Clara", "Daniel", "Elena", "Farid", "Grace", "Hannah",
    "Idris", "Jonas", "Karim", "Leila", "Marta", "Noor", "Oscar", "Priya",
    "Quentin", "Rosa", "Samir", "Tomas", "Ursula", "Viktor", "Wanda", "Yara",
    "Zoe", "Adam", "Bilal", "Celia", "Dmitri", "Esther", "Felix", "Gita",
    "Hugo", "Ines", "Jamal", "Kasia", "Lorenzo", "Mira", "Nadia", "Omar",
    "Pia", "Rafael", "Selma", "Theo", "Uma", "Vera", "Wesley", "Xenia",
    "Yusuf", "Zara", "Amira", "Bruno", "Camille", "Devi", "Eli", "Fiona",
    "Gabriel", "Hana", "Ivan", "Jade", "Kofi", "Lucia", "Mateo", "Nina",
    "Otto", "Paloma", "Rania", "Sofia", "Tariq", "Ulla", "Vasco", "Wren",
    "Yasmin", "Zeno", "Alina", "Bodhi", "Ciara", "Dara", "Emre", "Freya",
)

ROLES: Tuple[str, ...] = (
    "caregiver",
    "partner",
    "sibling",
    "clinician",
    "friend",
    "neighbour",
    "child",
    "therapist",
)

LOCATIONS: Tuple[str, ...] = ("bedroom", "kitchen", "living_room", "clinic", "garden", "outdoors")

OBJECTS: Tuple[str, ...] = (
    "blanket", "pillow", "wheelchair", "headrest", "footrest", "cushion", "duvet",
    "curtain", "window", "lamp", "radio", "television", "tablet", "phone", "charger",
    "photograph", "album", "letter", "notebook", "pen", "glasses", "hearing_aid",
    "straw", "beaker", "mug", "teapot", "kettle", "tea", "coffee", "water", "juice",
    "soup", "yoghurt", "toast", "porridge", "banana", "biscuit", "honey", "salt",
    "ventilator", "suction_tube", "nebuliser", "oxygen_mask", "feeding_tube",
    "syringe", "pill_box", "baclofen", "gabapentin", "paracetamol", "morphine",
    "riluzole", "eye_drops", "moisturiser", "sling", "splint", "hoist", "commode",
    "shower_chair", "towel", "flannel", "toothbrush", "comb", "razor", "nail_file",
    "slippers", "cardigan", "scarf", "socks", "blanket_throw", "footstool",
    "armchair", "bed_rail", "mattress", "call_button", "switch", "mount", "tripod",
    "eye_tracker", "keyboard", "screen", "speaker", "headphones", "remote_control",
    "clock", "calendar", "whiteboard", "postcard", "bookshelf", "novel", "newspaper",
    "crossword", "jigsaw", "chess_set", "radio_play", "podcast", "playlist",
    "guitar", "piano", "birdfeeder", "rosebush", "tomato_plant", "herb_pot",
    "watering_can", "greenhouse", "bench", "gate", "path", "fence", "shed",
    "compost", "lawn", "hedge", "wheelbarrow", "gloves", "hat", "umbrella",
    "raincoat", "boots", "walking_frame", "ramp", "doorbell", "post_box",
    "car", "minibus", "seatbelt", "map", "ticket", "wallet", "keys", "bag",
    "cat", "dog", "goldfish", "canary", "rabbit", "birdbath", "squirrel",
    "kitchen_timer", "recipe", "apron", "bowl", "spoon", "plate", "napkin",
    "blender", "microwave", "fridge", "freezer", "cupboard", "drawer", "shelf",
    "thermostat", "radiator", "fan", "humidifier", "air_filter", "blind",
    "candle", "vase", "mirror", "rug", "doormat", "painting", "sculpture",
    "grandchild_drawing", "wedding_ring", "watch", "bracelet", "locket",
    "medal", "trophy", "certificate", "diary", "passport", "bank_card",
    "hot_water_bottle", "ice_pack", "thermometer", "blood_pressure_cuff",
    "pulse_oximeter", "wheelchair_tray", "cup_holder", "seat_belt_pad",
    "communication_board", "alphabet_chart", "picture_card", "symbol_page",
    "buzzer", "pager", "intercom", "baby_monitor", "night_light", "torch",
)

ACTIVITIES: Tuple[str, ...] = (
    "breakfast", "lunch", "dinner", "snack", "tea_break", "medication_round",
    "physiotherapy", "speech_therapy", "occupational_therapy", "hydrotherapy",
    "stretching", "standing_frame", "repositioning", "transfer", "hoisting",
    "washing", "shower", "bed_bath", "shaving", "hair_wash", "teeth_cleaning",
    "dressing", "undressing", "toileting", "catheter_care", "skin_check",
    "suctioning", "nebuliser_session", "breathing_exercise", "cough_assist",
    "nap", "night_settle", "morning_routine", "evening_routine", "turning",
    "video_call", "phone_call", "visit", "outing", "hospital_appointment",
    "gp_appointment", "clinic_review", "blood_test", "scan", "swallow_study",
    "wheelchair_clinic", "equipment_review", "home_visit", "carer_handover",
    "shopping", "post_office", "bank_visit", "library_trip", "cafe_visit",
    "park_walk", "garden_sitting", "birdwatching", "gardening", "watering",
    "weeding", "planting", "harvesting", "pruning", "composting", "greenhouse_work",
    "reading", "audiobook", "podcast_listening", "music_listening", "radio_show",
    "television", "film_night", "football_match", "quiz_show", "documentary",
    "crossword", "jigsaw", "chess_game", "card_game", "board_game", "sudoku",
    "letter_writing", "email", "messaging", "photo_sorting", "album_making",
    "storytelling", "reminiscing", "prayer", "meditation", "mindfulness",
    "singing", "humming", "piano_listening", "choir_broadcast", "concert_stream",
    "grandchild_visit", "birthday", "anniversary", "christmas", "eid", "diwali",
    "holiday_planning", "travel", "day_trip", "seaside", "countryside_drive",
    "knitting_watching", "baking_watching", "cooking_together", "recipe_choosing",
    "pet_feeding", "dog_walk_watching", "cat_grooming", "fish_feeding",
    "news_catchup", "weather_check", "diary_update", "calendar_review",
    "care_plan_discussion", "medication_review", "symptom_report", "pain_review",
    "mood_check", "sleep_review", "appetite_check", "weight_check",
    "communication_practice", "device_calibration", "switch_setup", "voice_banking",
    "message_banking", "vocabulary_update", "page_customisation", "training_session",
    "peer_group", "support_group", "advocacy_meeting", "research_visit",
    "volunteering", "committee_call", "fundraising", "campaign_letter",
    "gardening_club", "book_club", "film_club", "music_group", "art_class",
    "photography", "painting_session", "pottery_watching", "craft_fair",
    "market_visit", "garden_centre", "museum_trip", "gallery_visit", "theatre",
    "cinema", "sports_match", "swimming_watching", "walking_group", "picnic",
)

ABSTRACT_STATES: Tuple[str, ...] = (
    "pain", "fatigue", "thirst", "hunger", "nausea", "breathlessness",
    "discomfort", "stiffness", "cramp", "itch", "cold", "heat",
    "happiness", "sadness", "frustration", "anxiety", "boredom", "loneliness",
    "relief", "gratitude",
)

IDIOLECT_FRAMES: Tuple[str, ...] = (
    "if you would be so kind",
    "no rush at all",
    "just a small one",
    "same as yesterday",
    "when you get a minute",
    "that would be lovely",
    "I am alright, honestly",
    "give it a moment",
)

DIAGNOSES = {
    "ALS": ("ALS (bulbar onset)", "ALS (limb onset)", "ALS (advanced)"),
    "CP": ("Cerebral palsy, GMFCS IV", "Cerebral palsy, GMFCS V"),
    "STROKE": ("Brainstem stroke (locked-in)", "Brainstem infarction"),
}
