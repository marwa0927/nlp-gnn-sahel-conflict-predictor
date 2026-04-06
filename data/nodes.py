"""
Shared source of truth: 25 Sahel city nodes covering known conflict corridors
across Mali, Niger, Burkina Faso, Chad, and Mauritania.
"""

NODES = [
    # ── Mali (7) ──────────────────────────────────────────────────────────────
    {"id": "bamako",    "lat": 12.65, "lon": -8.00,  "country": "ML"},
    {"id": "mopti",     "lat": 14.49, "lon": -4.20,  "country": "ML"},
    {"id": "gao",       "lat": 16.27, "lon": -0.04,  "country": "ML"},
    {"id": "timbuktu",  "lat": 16.77, "lon": -3.01,  "country": "ML"},
    {"id": "kidal",     "lat": 18.44, "lon":  1.41,  "country": "ML"},
    {"id": "menaka",    "lat": 15.92, "lon":  2.40,  "country": "ML"},
    {"id": "ansongo",   "lat": 15.67, "lon":  0.50,  "country": "ML"},

    # ── Niger (7) ─────────────────────────────────────────────────────────────
    {"id": "niamey",    "lat": 13.51, "lon":  2.11,  "country": "NI"},
    {"id": "agadez",    "lat": 16.97, "lon":  7.99,  "country": "NI"},
    {"id": "zinder",    "lat": 13.80, "lon":  8.99,  "country": "NI"},
    {"id": "tahoua",    "lat": 14.89, "lon":  5.27,  "country": "NI"},
    {"id": "diffa",     "lat": 13.32, "lon": 12.61,  "country": "NI"},
    {"id": "tillaberi", "lat": 14.21, "lon":  1.45,  "country": "NI"},
    {"id": "maradi",    "lat": 13.50, "lon":  7.10,  "country": "NI"},

    # ── Burkina Faso (5) ──────────────────────────────────────────────────────
    {"id": "ouagadougou",    "lat": 12.37, "lon": -1.53, "country": "UV"},
    {"id": "dori",           "lat": 14.03, "lon": -0.03, "country": "UV"},
    {"id": "djibo",          "lat": 14.10, "lon": -1.63, "country": "UV"},
    {"id": "bobo_dioulasso", "lat": 11.18, "lon": -4.30, "country": "UV"},
    {"id": "kaya",           "lat": 13.09, "lon": -1.08, "country": "UV"},

    # ── Chad (4) ──────────────────────────────────────────────────────────────
    {"id": "ndjamena", "lat": 12.11, "lon": 15.04, "country": "CD"},
    {"id": "abeche",   "lat": 13.83, "lon": 20.83, "country": "CD"},
    {"id": "mao",      "lat": 14.12, "lon": 15.31, "country": "CD"},
    {"id": "mongo",    "lat": 12.18, "lon": 18.69, "country": "CD"},

    # ── Mauritania (2) ────────────────────────────────────────────────────────
    {"id": "nouakchott", "lat": 18.08, "lon": -15.97, "country": "MR"},
    {"id": "nema",       "lat": 16.62, "lon":  -7.27, "country": "MR"},
]

NODES_DICT = {n["id"]: n for n in NODES}
