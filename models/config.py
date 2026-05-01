# analysis/config.py
#
# Model configuration and variable definitions.
# This module must contain ONLY constants and simple data structures.
# It must NOT import from tvp/, analysis/, or apps/.

# GVAR VARIABLES
START_DATE = "1997-01-01"
END_DATE = "2025-07-01"
# END_DATE = "2025-10-01"

# ----- 2 Choose variables and estimation window
DOMESTIC_VARS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
EXTERNAL_VARS = ["ENSO", "US_GDP_YoY", "CHN_GDP_YoY", "COMMODITY_YoY"]
EXTERNAL_VARS.append("PRITHVI_HEAT_STD")
EXTERNAL_VARS.append("PRITHVI_MOISTURE_STD")
EXTERNAL_VARS.append("PRITHVI_MOISTURE_EXTENT")
EXTERNAL_VARS.append("PRITHVI_HEAT_EXTENT")
EXTERNAL_VARS += [
    "CPI_YoY_annual",
    "FX_YoY_annual",
    "COMMODITY_AGR_YoY",
]
# EXTERNAL_VARS = ["ENSO"]

# ----- 5 Estimate VARX per country
FOREIGN_VARS = [f"{v}_star" for v in DOMESTIC_VARS]

# ----- RUN configuration
# START = "2005-01-01"
# END   = "2024-10-01"
START = "1997-01-01"
END   = "2025-10-01"

# ----- GVAR configuration
KalmanConfig = {
    # --- Estimation mode
    "use_TVP": False,          # False → fixed VAR
    "forgetting_lambda": 0.98,
    "measurement_noise_R": 1.0,

    # --- Shock handling
    "shock_scaling": "std",   # {"unit", "std", "percentile"}
    "shock_percentile": 84,
    "shock_duration_q": 1,

    # --- Regime selection
    "irf_regime": "typical",  # {"last", "typical", "stress", "custom"}
    "custom_t0": None,

    # --- Heat variable handling
    "heat_transform": "intensity",  # {"binary", "intensity", "zscore"}
    "heat_threshold": None,

    # --- Safety / damping
    "cap_eigenvalues": True,
    "max_eigenvalue": 0.98,

    # --- Reporting
    "label_units": "pp",      # {"pp", "%"}
}

# ----- COUNTRIES
# Countries (IMF codes)
ISO3_TO_IMF_NAME = {
    "MEX": "Mexico",
    "COL": "Colombia",
    "PER": "Peru",
    "BRA": "Brazil",
    "CHL": "Chile",
    "EGY": "Egypt",
    "KEN": "Kenya",
    "ZAF": "South Africa",
    "IND": "India",
    "IDN": "Indonesia",
    "THA": "Thailand",
    "PHL": "Philippines",
    "AUS": "Australia",
    "ESP": "Spain",
    "ECU": "Ecuador",
    "BHS": "Bahamas",
    "DOM": "Dominican Rep.", # "Dominican Republic"
    "JAM": "Jamaica",
    "PRY": "Paraguay",
    "NGA": "Nigeria",
    "PAK": "Pakistan",
    "VNM": "Vietnam",
    "LKA": "Sri Lanka",
    "TUN": "Tunisia",
    "MAR": "Morocco",
}

ISO3_TO_IMF_NAME_FULL = {
    "AFG": "Afghanistan",
    "ALB": "Albania",
    "DZA": "Algeria",
    "AND": "Andorra",
    "AGO": "Angola",
    "ATG": "Antigua and Barbuda",
    "ARG": "Argentina",
    "ARM": "Armenia",
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BHS": "Bahamas",
    "BHR": "Bahrain",
    "BGD": "Bangladesh",
    "BRB": "Barbados",
    "BLR": "Belarus",
    "BEL": "Belgium",
    "BLZ": "Belize",
    "BEN": "Benin",
    "BTN": "Bhutan",
    "BOL": "Bolivia",
    "BIH": "Bosnia and Herzegovina",
    "BWA": "Botswana",
    "BRA": "Brazil",
    "BRN": "Brunei Darussalam",
    "BGR": "Bulgaria",
    "BFA": "Burkina Faso",
    "BDI": "Burundi",
    "CPV": "Cabo Verde",
    "KHM": "Cambodia",
    "CMR": "Cameroon",
    "CAN": "Canada",
    "CAF": "Central African Republic",
    "TCD": "Chad",
    "CHL": "Chile",
    "CHN": "China",
    "COL": "Colombia",
    "COM": "Comoros",
    "COG": "Congo, Republic of",
    "COD": "Congo, Democratic Republic of the",
    "CRI": "Costa Rica",
    "CIV": "Côte d'Ivoire",
    "HRV": "Croatia",
    "CUB": "Cuba",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DNK": "Denmark",
    "DJI": "Djibouti",
    "DMA": "Dominica",
    "DOM": "Dominican Republic",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "SLV": "El Salvador",
    "GNQ": "Equatorial Guinea",
    "ERI": "Eritrea",
    "EST": "Estonia",
    "SWZ": "Eswatini",
    "ETH": "Ethiopia",
    "FJI": "Fiji",
    "FIN": "Finland",
    "FRA": "France",
    "GAB": "Gabon",
    "GMB": "Gambia",
    "GEO": "Georgia",
    "DEU": "Germany",
    "GHA": "Ghana",
    "GRC": "Greece",
    "GRD": "Grenada",
    "GTM": "Guatemala",
    "GIN": "Guinea",
    "GNB": "Guinea-Bissau",
    "GUY": "Guyana",
    "HTI": "Haiti",
    "HND": "Honduras",
    "HUN": "Hungary",
    "ISL": "Iceland",
    "IND": "India",
    "IDN": "Indonesia",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "IRL": "Ireland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JPN": "Japan",
    "JOR": "Jordan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KIR": "Kiribati",
    "KWT": "Kuwait",
    "KGZ": "Kyrgyz Republic",
    "LAO": "Lao PDR",
    "LVA": "Latvia",
    "LBN": "Lebanon",
    "LSO": "Lesotho",
    "LBR": "Liberia",
    "LBY": "Libya",
    "LIE": "Liechtenstein",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MDG": "Madagascar",
    "MWI": "Malawi",
    "MYS": "Malaysia",
    "MDV": "Maldives",
    "MLI": "Mali",
    "MLT": "Malta",
    "MHL": "Marshall Islands",
    "MRT": "Mauritania",
    "MUS": "Mauritius",
    "MEX": "Mexico",
    "FSM": "Micronesia",
    "MDA": "Moldova",
    "MCO": "Monaco",
    "MNG": "Mongolia",
    "MNE": "Montenegro",
    "MAR": "Morocco",
    "MOZ": "Mozambique",
    "MMR": "Myanmar",
    "NAM": "Namibia",
    "NRU": "Nauru",
    "NPL": "Nepal",
    "NLD": "Netherlands",
    "NZL": "New Zealand",
    "NIC": "Nicaragua",
    "NER": "Niger",
    "NGA": "Nigeria",
    "PRK": "Korea, North",
    "MKD": "North Macedonia",
    "NOR": "Norway",
    "OMN": "Oman",
    "PAK": "Pakistan",
    "PLW": "Palau",
    "PAN": "Panama",
    "PNG": "Papua New Guinea",
    "PRY": "Paraguay",
    "PER": "Peru",
    "PHL": "Philippines",
    "POL": "Poland",
    "PRT": "Portugal",
    "QAT": "Qatar",
    "ROU": "Romania",
    "RUS": "Russia",
    "RWA": "Rwanda",
    "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia",
    "VCT": "Saint Vincent and the Grenadines",
    "WSM": "Samoa",
    "SMR": "San Marino",
    "STP": "Sao Tome and Principe",
    "SAU": "Saudi Arabia",
    "SEN": "Senegal",
    "SRB": "Serbia",
    "SYC": "Seychelles",
    "SLE": "Sierra Leone",
    "SGP": "Singapore",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SLB": "Solomon Islands",
    "SOM": "Somalia",
    "ZAF": "South Africa",
    "KOR": "South Korea",
    "SSD": "South Sudan",
    "ESP": "Spain",
    "LKA": "Sri Lanka",
    "SDN": "Sudan",
    "SUR": "Suriname",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "SYR": "Syria",
    "TJK": "Tajikistan",
    "TZA": "Tanzania",
    "THA": "Thailand",
    "TLS": "Timor-Leste",
    "TGO": "Togo",
    "TON": "Tonga",
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "TKM": "Turkmenistan",
    "TUV": "Tuvalu",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "ARE": "United Arab Emirates",
    "GBR": "United Kingdom",
    "USA": "United States",
    "URY": "Uruguay",
    "UZB": "Uzbekistan",
    "VUT": "Vanuatu",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
    "YEM": "Yemen",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
}

ISO3_TO_IMF_NAME = ISO3_TO_IMF_NAME_FULL

COUNTRIES = ISO3_TO_IMF_NAME

COUNTRY_CODES = list(ISO3_TO_IMF_NAME.keys())
COUNTRY_NAMES = list(ISO3_TO_IMF_NAME.values())

ISO3 = list(ISO3_TO_IMF_NAME.keys())

IMF_EX_NAMES = ["Yemen, People's Democratic Republic of",
       'Union of Soviet Socialist Republics (USSR)',
       'Yemen Arab Republic', 'South African Common Customs Area (SACCA)',
       'German Democratic Republic', 'Czechoslovakia',
       'Yugoslavia, Socialist Federal Republic of', 'Belgium-Luxembourg',
       'Mexico', 'Cyprus', 'Iceland', 'United States', 'New Zealand',
       "China, People's Republic of", 'Italy', 'Dominican Republic',
       'Uzbekistan, Republic of', 'Rwanda', 'Senegal', 'Ghana',
       'Kyrgyz Republic', 'Malawi', 'Iran, Islamic Republic of',
       'Nauru, Republic of', 'Micronesia, Federated States of', 'Jamaica',
       'Marshall Islands, Republic of the', 'United Kingdom', 'Lebanon',
       'Indonesia', 'Pakistan', 'Tanzania, United Republic of',
       'El Salvador', 'Greenland', 'Yemen, Republic of',
       'Russian Federation', 'Tuvalu', 'Ukraine', 'Saudi Arabia', 'Guam',
       'Sierra Leone', 'Iraq', 'Vietnam', 'Spain', 'France',
       'Madagascar, Republic of', 'Myanmar', 'Czech Republic',
       'Congo, Republic of', 'Holy See', 'Niger',
       'Aruba, Kingdom of the Netherlands',
       'Afghanistan, Islamic Republic of', "Côte d'Ivoire", 'Tunisia',
       'Portugal', 'Gambia, The', 'Azerbaijan, Republic of', 'Malta',
       'Bangladesh', 'North Macedonia, Republic of', 'Kenya', 'Finland',
       'Uganda', 'Belarus, Republic of', 'Benin', 'Montenegro',
       'Latvia, Republic of', 'Korea, Republic of', 'Burkina Faso',
       'Sudan', 'Lithuania, Republic of', 'Zimbabwe', 'Singapore',
       'Israel', 'Greece', 'Brazil', 'Japan', 'Chile',
       'Montserrat, United Kingdom-British Overseas Territory', 'Tonga',
       "Macao Special Administrative Region, People's Republic of China",
       'Antigua and Barbuda', 'Hungary', 'Serbia, Republic of',
       'Falkland Islands (Malvinas)', 'Guyana', 'Nepal', 'Faroe Islands',
       'Ireland', 'Advanced Economies', 'Bahamas, The', 'Zambia',
       'Bosnia and Herzegovina', 'Emerging and Developing Asia',
       'Armenia, Republic of',
       'São Tomé and Príncipe, Democratic Republic of', 'Africa',
       'Germany', 'Togo', 'Cameroon', 'Emerging and Developing Europe',
       'Other Countries n.i.e.', 'Qatar', 'Middle East', 'Switzerland',
       'Tajikistan, Republic of', 'World',
       'Middle East, North Africa, Afghanistan, and Pakistan',
       'Latin America and the Caribbean (LAC)', 'Kuwait', 'CIS',
       'European Union (EU)', 'Oman', 'Kiribati',
       'Sub-Saharan Africa (SSA)', 'Central African Republic',
       'New Caledonia', 'Euro Area (EA)', 'Luxembourg', 'Cuba',
       'Cabo Verde', 'Middle East and Central Asia',
       'Netherlands Antilles', 'Brunei Darussalam', 'American Samoa',
       'Kosovo, Republic of', 'Fiji, Republic of',
       'Equatorial Guinea, Republic of', 'Liberia', 'Djibouti',
       'Moldova, Republic of', 'Canada',
       'Anguilla, United Kingdom-British Overseas Territory', 'Sweden',
       'Seychelles', 'San Marino, Republic of', 'Costa Rica', 'Mali',
       'Sint Maarten, Kingdom of the Netherlands',
       "Hong Kong Special Administrative Region, People's Republic of China",
       'Croatia, Republic of', "Lao People's Democratic Republic",
       'Egypt, Arab Republic of', 'Belize', 'Palau, Republic of',
       'Maldives', 'Thailand', 'Eritrea, The State of', 'Denmark',
       'Algeria', 'Guinea-Bissau', 'Australia', 'Georgia',
       'Poland, Republic of', 'Congo, Democratic Republic of the', 'Chad',
       'Morocco', 'Curaçao, Kingdom of the Netherlands',
       'Bahrain, Kingdom of', 'Emerging Market and Developing Economies',
       'Netherlands, The', 'Bhutan', 'Angola', 'Estonia, Republic of',
       'Grenada', 'India', 'Vanuatu', 'Bermuda', 'Turkmenistan', 'Samoa',
       'St. Kitts and Nevis', 'French Polynesia', 'Bulgaria', 'Argentina',
       'Bolivia', 'Mozambique, Republic of', 'Comoros, Union of the',
       'Gibraltar', 'Libya', 'West Bank and Gaza',
       'St. Vincent and the Grenadines', 'Trinidad and Tobago',
       'Slovenia, Republic of', 'Belgium', 'Slovak Republic', 'Sri Lanka',
       'Kazakhstan, Republic of', 'Papua New Guinea', 'Austria',
       'Dominica', 'Paraguay', 'Syrian Arab Republic',
       'EMDEs by Source of Export Earnings: Fuel', 'Nicaragua', 'Europe',
       'EMDEs by Source of Export Earnings: Nonfuel', 'Nigeria',
       'Uruguay', 'Albania', 'Eswatini, Kingdom of', 'Malaysia',
       'South Sudan, Republic of', 'Guinea', 'Romania', 'Somalia',
       'Cambodia', 'Botswana', 'Ecuador', 'Mauritius',
       'Timor-Leste, Democratic Republic of', 'Mongolia', 'Suriname',
       'United Arab Emirates', 'St. Lucia', 'Panama',
       'Ethiopia, The Federal Democratic Republic of', 'Barbados',
       'Lesotho, Kingdom of', 'Mauritania, Islamic Republic of',
       'Burundi', 'Solomon Islands', 'South Africa',
       "Korea, Democratic People's Republic of", 'Peru',
       'Serbia and Montenegro', 'Colombia',
       'Venezuela, República Bolivariana de', 'Norway', 'Guatemala',
       'Türkiye, Republic of', 'Jordan', 'Namibia', 'Honduras', 'Haiti',
       'Philippines', 'Gabon']

import country_converter as coco
cc = coco.CountryConverter()
# 1. Convert the list of names to ISO3 codes
iso3_codes = cc.convert(names=IMF_EX_NAMES, to='ISO3')
# 2. Map ISO3 as keys and IMF names as values
# zip(keys, values) creates the pairs
iso3_to_imf_EX_dict = {}
for iso, name in zip(iso3_codes, IMF_EX_NAMES):
    # If coco returns a list, take the first element; otherwise use the string
    clean_iso = iso[0] if isinstance(iso, list) else iso
    if clean_iso != 'not found':
        iso3_to_imf_EX_dict[clean_iso] = name

REGIONMASK_TO_ISO3 = {
    "BR": "BRA",
    "CL": "CHL",
    "CO": "COL",
    "MX": "MEX",
    "PE": "PER",
    "PH": "PHL",
    "TH": "THA",
    "ZA": "ZAF",
    "KE": "KEN",
    "EG": "EGY",
    "IND": "IND",  # India
    "INDO": "IDN",  # Indonesia
    "AU": "AUS",
    "E": "ESP", # ES
    "EC": "ECU",
    "BS": "BHS",
    "DO": "DOM",
    "J": "JAM", # JM
    "PY": "PRY",
    "NG": "NGA",
    "PK": "PAK",
    "VN": "VNM",
    "LK": "LKA",
    "TN": "TUN",
    "MA": "MAR",
}

ISO3_TO_EURO_RATE = {
    "AUT": 13.7603,     # Austrian schilling
    "BEL": 40.3399,     # Belgian franc
    "HRV": 7.53450,     # Croatian kuna
    "CYP": 0.585274,    # Cypriot pound
    "EST": 15.6466,     # Estonian kroon
    "FIN": 5.94573,     # Finnish markka
    "FRA": 6.55957,     # French franc
    "GRC": 340.750,     # Greek drachma
    "IRL": 0.787564,    # Irish pound
    "ITA": 1936.27,     # Italian lira
    "LVA": 0.702804,    # Latvian lats
    "LTU": 3.45280,     # Lithuanian litas
    "LUX": 40.3399,     # Luxembourg franc
    "MLT": 0.429300,    # Maltese lira
    "NLD": 2.20371,     # Dutch guilder
    "PRT": 200.482,     # Portuguese escudo
    "SVK": 30.1260,     # Slovak koruna
    "SVN": 239.640,     # Slovenian tolar
    "ESP": 166.386      # Spanish peseta
}

ISO3_TO_EURO_CONVERSION_QTR = {
    # Founding euro adopters (book money 1999-Q1, cash 2002-Q1)
    "AUT": "1999-Q1",
    "BEL": "1999-Q1",
    "FIN": "1999-Q1",
    "FRA": "1999-Q1",
    "IRL": "1999-Q1",
    "ITA": "1999-Q1",
    "LUX": "1999-Q1",
    "NLD": "1999-Q1",
    "PRT": "1999-Q1",
    "ESP": "1999-Q1",

    # Later adopters
    "GRC": "2001-Q1",
    "SVN": "2007-Q1",
    "CYP": "2008-Q1",
    "MLT": "2008-Q1",
    "SVK": "2009-Q1",
    "EST": "2011-Q1",
    "LVA": "2014-Q1",
    "LTU": "2015-Q1",
    "HRV": "2023-Q1"
}

# ----- TEMPORARY
flag_temp = False
if flag_temp:
    ISO3_no_GDP = [
        "AGO","ATG","BGD","BRB","BLZ","BEN","BTN","BFA","BDI","KHM",
        "CAF","TCD","COG","COD","CIV","DJI","DMA","GAB","GRD","GIN",
        "GNB","GUY","HTI","IRQ","KIR","KWT","LBN","LBR","LBY","MWI",
        "MLI","MMR","NPL","NER","OMN","PAN","PNG","SLE","SLB","SOM",
        "SDN","SUR","TGO","TON","TUN","TUV","ARE","VUT","VNM","ZMB",
        "ZWE"
    ]

    # GVAR Toolbox:
    ISO3_GVAR_Toolbox = [
    "ARG",  # Argentina
    "AUS",  # Australia
    "AUT",  # Austria
    "BEL",  # Belgium
    "BRA",  # Brazil
    "CAN",  # Canada
    "CHL",  # Chile
    "CHN",  # China
    "COL",  # Colombia
    "EGY",  # Egypt
    "FIN",  # Finland
    "FRA",  # France
    "DEU",  # Germany
    "IND",  # India
    "IDN",  # Indonesia
    "ITA",  # Italy
    "JPN",  # Japan
    "KEN",  # Kenya
    "KOR",  # Korea (South Korea)
    "MYS",  # Malaysia
    "MEX",  # Mexico
    "NLD",  # Netherlands
    "NZL",  # New Zealand
    "NOR",  # Norway
    "PER",  # Peru
    "PHL",  # Philippines
    "SAU",  # Saudi Arabia
    "SGP",  # Singapore
    "ZAF",  # South Africa
    "ESP",  # Spain
    "SWE",  # Sweden
    "CHE",  # Switzerland
    "THA",  # Thailand
    "TUR",  # Turkey
    "GBR",  # United Kingdom
    "USA",  # United States
    ]

    # Tongai
    ISO3_DM = [
    "AUS",  # Australia
    "ESP",  # Spain
    ]
    ISO3_EM = [
    "ECU",  # Ecuador
    "BHS",  # Bahamas
    "DOM",  # Dominican Republic
    "JAM",  # Jamaica
    "PRY",  # Paraguay
    "NGA",  # Nigeria
    "PAK",  # Pakistan
    "VNM",  # Vietnam
    "LKA",  # Sri Lanka
    "TUN",  # Tunisia
    "MAR",  # Morocco
    ]

    ISO3_important_no_GDP = list(set(ISO3_no_GDP) &
        (set(ISO3_GVAR_Toolbox) | set(ISO3_DM) | set(ISO3_EM)))

    countries_important_no_GDP = []
    for iso in ISO3_important_no_GDP:
        countries_important_no_GDP.append(ISO3_TO_IMF_NAME_FULL[iso])

    print("Countries with CPI, EX, FX from IMF but not GDP")
    print(countries_important_no_GDP)
    print(ISO3_important_no_GDP)

    ISO3_complete = [
        "ALB","DZA","ARG","AUS","AUT","BEL","BOL","BIH","BWA","BRA",
        "BRN","BGR","CPV","CMR","CAN","CHL","COL","CRI","CYP","CZE",
        "DNK","DOM","ECU","SLV","FIN","FRA","GEO","DEU","GHA","GRC",
        "GTM","HND","HUN","ISL","IND","IDN","IRL","ISR","ITA","JAM",
        "JPN","JOR","KEN","KGZ","LUX","MYS","MDV","MLT","MUS","MEX",
        "MNG","MNE","MAR","NAM","NZL","NIC","NGA","NOR","PAK","PRY",
        "PER","PHL","PRT","QAT","ROU","RWA","WSM","SAU","SEN","SYC",
        "SGP","ZAF","ESP","LKA","SWE","CHE","THA","TTO","UGA","UKR",
        "GBR","USA","URY"
    ]
    ISO3_with_GDP = list(set(ISO3) - set(ISO3_no_GDP))
    ISO3_EM_DM_no_GDP = list(set(ISO3_DM + ISO3_EM) - set(ISO3_with_GDP))
    print(ISO3_EM_DM_no_GDP)

    ISO3_EM_DM_incomplete = list(set(ISO3_DM + ISO3_EM) - set(ISO3_complete))
    print(ISO3_EM_DM_incomplete)

    exit()