"""
Botanical Knowledge Base and Disease Care Recommendation Engine.
Contains comprehensive diagnostic profiles, symptoms, causes, immediate actions,
treatments (organic and chemical), and preventative care tips for all 16 classes.
"""

import json
from pathlib import Path

DISEASE_KNOWLEDGE_BASE = {
    "Aloe_Anthracnose": {
        "class_id": "Aloe_Anthracnose",
        "plant_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "condition_name": "Anthracnose Leaf Spot",
        "is_healthy": False,
        "severity": "Moderate",
        "category": "Fungal Infection",
        "summary": "A fungal disease caused by Colletotrichum species that produces sunken, dark necrotic lesions on succulent aloe leaves.",
        "symptoms": [
            "Round to oval sunken lesions with dark brown or black centers",
            "Water-soaked margins surrounding the spots",
            "Lesions coalescing into large rotted sunken areas on older leaves",
            "Spore masses turning pinkish or orange under high humidity"
        ],
        "causes": [
            "Prolonged leaf wetness from overhead watering or misting",
            "High humidity coupled with poor air circulation",
            "Fungal spores splashing from contaminated soil or adjacent infected plants"
        ],
        "immediate_actions": [
            "Isolate the plant immediately from other succulents to prevent spore spread.",
            "Sterilize shears with 70% isopropyl alcohol and excise severely infected leaf sections near the base.",
            "Cease all foliar misting and keep leaves completely dry."
        ],
        "treatment": {
            "organic": [
                "Dust freshly cut leaf surfaces with organic sulfur powder or ground cinnamon (natural antifungal sealant).",
                "Apply cold-pressed organic neem oil spray (1 tsp neem + 1/2 tsp mild castile soap per liter of warm water)."
            ],
            "chemical": [
                "Apply broad-spectrum liquid copper fungicide spray according to label instructions.",
                "In severe recurrent cases, apply a systemic fungicide containing chlorothalonil or thiophanate-methyl."
            ]
        },
        "prevention": [
            "Water exclusively at soil level; never pour water into the central leaf rosette.",
            "Allow the potting mix to dry out 100% between waterings.",
            "Ensure bright, indirect sunlight and good room ventilation."
        ]
    },

    "Aloe_Healthy": {
        "class_id": "Aloe_Healthy",
        "plant_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "condition_name": "Healthy Baseline",
        "is_healthy": True,
        "severity": "None",
        "category": "Healthy",
        "summary": "Vibrant, plump succulent leaves with firm green foliage and no active pathological infections.",
        "symptoms": [
            "Firm, erect, fleshy green to grey-green succulent leaves",
            "Smooth leaf surface without sunken spots, rot, or lesions",
            "Serrated edges with pale teeth intact"
        ],
        "causes": [
            "Proper balance of gritty fast-draining substrate, infrequent soak watering, and ample sunlight."
        ],
        "immediate_actions": [
            "No remedial action needed! Your Aloe Vera is thriving."
        ],
        "treatment": {
            "organic": [
                "Maintain natural care regimen with occasional diluted kelp or worm casting tea in spring."
            ],
            "chemical": [
                "No chemical applications required."
            ]
        },
        "prevention": [
            "Follow the 'soak-and-dry' watering rule: water deeply only once every 2-3 weeks.",
            "Provide 6+ hours of bright indirect or filtered direct sunlight daily.",
            "Maintain cactus/succulent gritty soil with 50% perlite, pumice, or coarse sand."
        ]
    },

    "Aloe_LeafSpot": {
        "class_id": "Aloe_LeafSpot",
        "plant_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "condition_name": "Fungal/Bacterial Leaf Spot",
        "is_healthy": False,
        "severity": "Mild to Moderate",
        "category": "Fungal/Bacterial Infection",
        "summary": "Infection typically caused by Alternaria, Cercospora, or bacterial pathogens producing scattered circular blemishes across leaves.",
        "symptoms": [
            "Small circular or irregular reddish-brown to dark spots on leaf surfaces",
            "Chlorotic yellow halos encircling the blemishes",
            "Older lower leaves showing denser spot concentrations"
        ],
        "causes": [
            "Overhead watering allowing water droplets to linger on foliage",
            "Stagnant indoor air and warm, humid microclimates"
        ],
        "immediate_actions": [
            "Move plant to a dry, well-ventilated location with bright light.",
            "Prune off heavily spotted individual leaves with sanitized blades.",
            "Wipe surrounding foliage dry with clean paper towels."
        ],
        "treatment": {
            "organic": [
                "Spray with a solution of 1 tbsp baking soda and 1/2 tsp castile soap in 1 liter of water to alter leaf surface pH.",
                "Apply sulfur-based garden dust directly to lesions."
            ],
            "chemical": [
                "Apply fixed copper fungicide or mancozeb spray every 7-10 days until new growth emerges spot-free."
            ]
        },
        "prevention": [
            "Avoid splashing soil onto lower foliage during watering.",
            "Use pots with generous drainage holes and avoid saucer pooling.",
            "Space plants at least 15 cm apart to facilitate airflow."
        ]
    },

    "Aloe_Rust": {
        "class_id": "Aloe_Rust",
        "plant_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "condition_name": "Aloe Rust",
        "is_healthy": False,
        "severity": "Moderate",
        "category": "Fungal Infection",
        "summary": "Specialized fungal disease caused by Uromyces aloes that forms elevated brown/black pustules embedded in leaf tissue.",
        "symptoms": [
            "Small yellow or pale chlorotic spots that darken rapidly",
            "Elevated round pustules (uredinia) that turn dark reddish-brown to black",
            "Circular depressions formed as leaf tissue under pustules collapses",
            "Permanent black scars remaining on older succulent tissue"
        ],
        "causes": [
            "Spore dissemination through airborne moisture and humid environments",
            "Wet leaf surfaces and cool to warm temperatures (18-24°C)"
        ],
        "immediate_actions": [
            "Isolate the aloe immediately; rust spores are highly contagious to other Aloes and Gasterias.",
            "Prune infected leaves to remove sporulating pustules. Dispose in sealed trash; do NOT compost."
        ],
        "treatment": {
            "organic": [
                "Dust thoroughly with wettable sulfur powder.",
                "Wipe uninfected foliage with diluted potassium bicarbonate solution."
            ],
            "chemical": [
                "Apply systemic fungicides containing myclobutanil, propiconazole, or copper oxychloride.",
                "Repeat applications according to manufacturer instructions until no new pustules appear."
            ]
        },
        "prevention": [
            "Keep the plant in the sunniest, driest spot available.",
            "Never allow water to stand on leaves overnight.",
            "Sanitize all gardening tools with 10% bleach or rubbing alcohol after handling."
        ]
    },

    "Aloe_Sunburn": {
        "class_id": "Aloe_Sunburn",
        "plant_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "condition_name": "Sunburn / Photodamage",
        "is_healthy": False,
        "severity": "Mild to Moderate",
        "category": "Abiotic / Environmental Stress",
        "summary": "Tissue scorch caused by sudden exposure to intense direct sunlight, ultraviolet radiation, or heat stress.",
        "symptoms": [
            "Foliage turning reddish-brown, bronze, or orange-tinged",
            "Bleached white, grey, or straw-colored dried papery patches on top-facing leaf surfaces",
            "Leaves feeling thin, deflated, and curling inward"
        ],
        "causes": [
            "Abrupt transition from low indoor light to intense direct outdoor or south-facing window sun without hardening off",
            "Intense magnifying heat through unshaded window glass in summer"
        ],
        "immediate_actions": [
            "Relocate the aloe immediately into bright, filtered or dappled indirect sunlight.",
            "Do NOT overwater in panic—sunburnt roots cannot absorb excess water and will rot!"
        ],
        "treatment": {
            "organic": [
                "Allow the plant to naturally metabolize protective anthocyanins in gentle shade.",
                "New central shoots will grow out vivid green; severely crisp bleached patches can be trimmed once dry."
            ],
            "chemical": [
                "No chemicals needed. This is an abiotic physical injury, not an infection."
            ]
        },
        "prevention": [
            "Acclimate indoor aloes gradually over 10-14 days when shifting to brighter locations.",
            "Use sheer curtains on scorching south/west-facing window ledges during midday summer hours."
        ]
    },

    "Cactus_Dactylopius_Opuntia": {
        "class_id": "Cactus_Dactylopius_Opuntia",
        "plant_name": "Cactus (Opuntia / Prickly Pear)",
        "scientific_name": "Opuntia spp.",
        "condition_name": "Cochineal Scale Bug Infestation",
        "is_healthy": False,
        "severity": "Severe",
        "category": "Pest Infestation",
        "summary": "Infestation by Dactylopius scale insects that coat themselves in white, waxy, cottony protective filaments while sucking plant sap.",
        "symptoms": [
            "Fluffy, white cotton-like clusters congregating around areoles, spine clusters, and pad creases",
            "Crushing the white tufts yields a bright crimson-red pigment (carminic acid)",
            "Yellowing, chlorosis, and progressive shriveling of affected cactus pads",
            "Premature pad drop and stunted new growth"
        ],
        "causes": [
            "Introduction of infested new succulents without quarantine",
            "Warm, dry, still indoor environments favoring scale reproduction"
        ],
        "immediate_actions": [
            "Quarantine the cactus immediately away from all succulents and cacti.",
            "Dip cotton swabs in 70% isopropyl alcohol and dab directly onto white clusters to dissolve the wax and kill insects on contact.",
            "Use a soft toothbrush or tweezers to mechanically scrape off stubborn clusters."
        ],
        "treatment": {
            "organic": [
                "Spray thoroughly with insecticidal soap or pure cold-pressed neem oil emulsion, reaching into crevices.",
                "Repeat treatment every 5 to 7 days for at least 3-4 consecutive weeks to eliminate emerging crawler nymphs."
            ],
            "chemical": [
                "For severe infestations, apply a systemic insecticide containing acetamiprid or imidacloprid as a soil drench.",
                "Use pyrethrin-based contact aerosol sprays directly on localized colonies."
            ]
        },
        "prevention": [
            "Inspect newly acquired cacti meticulously with a magnifying lens before placing them with your collection.",
            "Provide strong air circulation and maintain regular foliage checks."
        ]
    },

    "Cactus_Healthy": {
        "class_id": "Cactus_Healthy",
        "plant_name": "Cactus",
        "scientific_name": "Cactaceae family",
        "condition_name": "Healthy Baseline",
        "is_healthy": True,
        "severity": "None",
        "category": "Healthy",
        "summary": "Firm, robust succulent cactus pads and stems with healthy areoles, intact spines, and zero pest infestations.",
        "symptoms": [
            "Rigid, plump, turgid green or bluish-green pad surfaces",
            "Clean areoles and strong spine development without discolouration",
            "No soft mushy spots, yellow halos, or cottony pest masses"
        ],
        "causes": [
            "Optimal desert-like environment: gritty mineral potting mix, abundant sunlight, and disciplined dry cycles."
        ],
        "immediate_actions": [
            "Continue current maintenance practices. Your cactus is in prime condition!"
        ],
        "treatment": {
            "organic": [
                "Feed once or twice in spring with dilute low-nitrogen cactus fertilizer."
            ],
            "chemical": [
                "No chemicals required."
            ]
        },
        "prevention": [
            "Keep in maximum bright sunlight (at least 6-8 hours daily).",
            "Water only when the substrate is 100% dry from top to bottom (once every 3-4 weeks in summer, 6-8 weeks in winter).",
            "Ensure pot has unrestricted bottom drainage."
        ]
    },

    "Money_Plant_Bacterial_wilt_disease": {
        "class_id": "Money_Plant_Bacterial_wilt_disease",
        "plant_name": "Money Plant (Pothos)",
        "scientific_name": "Epipremnum aureum",
        "condition_name": "Bacterial Wilt Disease",
        "is_healthy": False,
        "severity": "Severe",
        "category": "Bacterial Infection",
        "summary": "Lethal vascular bacterial infection (e.g. Ralstonia / Erwinia) clogging water-conducting xylem vessels, causing rapid collapse.",
        "symptoms": [
            "Sudden rapid drooping and wilting of vines despite the soil being damp or wet",
            "Water-soaked, dark translucent lesions along leaf petiole and main veins",
            "Soft, slimy, blackened rotting stems near the soil line",
            "Foul sour odor emanating from decomposing stem tissue"
        ],
        "causes": [
            "Waterlogged, poorly aerated potting soil",
            "Bacterial invasion through damaged root tips or pruning wounds",
            "Use of unsterilized shears or contaminated re-used potting mix"
        ],
        "immediate_actions": [
            "Quarantine immediately. Bacterial wilt is aggressive and spreads through contact and runoff.",
            "Cut away all collapsing, blackened vines well above the infected zone using sterilized shears.",
            "Discard heavily rotted root balls and contaminated soil in household waste (do NOT compost)."
        ],
        "treatment": {
            "organic": [
                "Take clean, asymptomatic tip cuttings (at least 2 nodes above any rot), sanitize in mild hydrogen peroxide (3% diluted 1:4 with water), and propagate in fresh clean water.",
                "Dust surviving cut nodes with sulfur or activated charcoal."
            ],
            "chemical": [
                "Drench soil of salvageable plants with agricultural copper bactericide (copper hydroxide).",
                "Disinfect pots and saucers with a 10% bleach solution before re-potting."
            ]
        },
        "prevention": [
            "Never allow money plants to sit in stagnant standing water inside outer decorative pots.",
            "Use an airy, chunky aroid potting mix (potting soil + perlite + orchid bark).",
            "Allow the top 2 inches of soil to dry completely before watering again."
        ]
    },

    "Money_Plant_Healthy": {
        "class_id": "Money_Plant_Healthy",
        "plant_name": "Money Plant (Pothos)",
        "scientific_name": "Epipremnum aureum",
        "condition_name": "Healthy Baseline",
        "is_healthy": True,
        "severity": "None",
        "category": "Healthy",
        "summary": "Lush, glossy heart-shaped foliage with strong turgor pressure, active vine development, and vivid green/variegated coloration.",
        "symptoms": [
            "Waxy, vibrant green or gold/white variegated leaves",
            "Firm, flexible stems with healthy aerial root nodes",
            "Uniform leaf expansion without curling, spots, or wilting"
        ],
        "causes": [
            "Balanced watering, medium to bright indirect lighting, and proper indoor ambient humidity."
        ],
        "immediate_actions": [
            "No corrective actions needed. Your Money Plant is thriving!"
        ],
        "treatment": {
            "organic": [
                "Wipe leaves periodically with a damp microfiber cloth to remove dust and maximize photosynthesis."
            ],
            "chemical": [
                "No chemicals required."
            ]
        },
        "prevention": [
            "Keep in bright indirect light for best variegation (avoid harsh midday direct sun).",
            "Water when the top 1-2 inches feel dry to the touch.",
            "Fertilize monthly during spring and summer with balanced houseplant liquid fertilizer at half strength."
        ]
    },

    "Money_Plant_Manganese_Toxicity": {
        "class_id": "Money_Plant_Manganese_Toxicity",
        "plant_name": "Money Plant (Pothos)",
        "scientific_name": "Epipremnum aureum",
        "condition_name": "Manganese Toxicity Disorder",
        "is_healthy": False,
        "severity": "Moderate",
        "category": "Nutritional / Soil Chemical Disorder",
        "summary": "Nutrient imbalance caused by excessive manganese availability in acidic soils (pH < 5.2) or over-fertilization, causing leaf necrosis.",
        "symptoms": [
            "Interveinal chlorosis (yellowing of leaf tissue between dark green veins) on older leaves",
            "Small, scattered dark brown to black necrotic spots with halos on mature leaves",
            "Crinkling, cupping, or necrotic margin scorching of affected foliage",
            "Stunted leaf development and premature drop of older foliage"
        ],
        "causes": [
            "Excessively low potting soil pH (acidic substrate dissolves excessive manganese into toxic soluble ions)",
            "Overuse of synthetic micronutrient fertilizers or high-phosphorus fertilizers that destabilize soil chemistry",
            "Poor soil aeration and prolonged waterlogging accelerating heavy metal uptake"
        ],
        "immediate_actions": [
            "Halt all chemical and synthetic fertilizer applications immediately.",
            "Thoroughly flush the root zone with copious volumes of distilled, reverse-osmosis, or dechlorinated water (3x pot volume) to leach excess mineral salts."
        ],
        "treatment": {
            "organic": [
                "Top-dress the soil with a small amount of dolomitic limestone or crushed eggshells to gently raise the pH toward 6.2-6.5.",
                "Repot into a fresh, balanced potting mix with organic compost, peat, and perlite if soil remains stubbornly acidic."
            ],
            "chemical": [
                "Test soil run-off pH with a digital pH meter; maintain target pH between 6.0 and 6.5 where manganese bioavailability is safe.",
                "Apply chelated iron (Fe-EDDHA) drench to outcompete and buffer excess manganese uptake."
            ]
        },
        "prevention": [
            "Always follow fertilizer dosing strictly: use half-strength dilution on indoor houseplants.",
            "Avoid using acidic peat-only substrates without limestone buffering.",
            "Repot annually to refresh exhausted potting media."
        ]
    },

    "Snake_Plant_Anthracnose": {
        "class_id": "Snake_Plant_Anthracnose",
        "plant_name": "Snake Plant (Sansevieria)",
        "scientific_name": "Dracaena trifasciata",
        "condition_name": "Anthracnose Leaf Lesion",
        "is_healthy": False,
        "severity": "Moderate to Severe",
        "category": "Fungal Infection",
        "summary": "Fungal infection by Colletotrichum sansevieriae producing unsightly sunken lesions that breach the thick cuticle of snake plant blades.",
        "symptoms": [
            "Sunken circular to elongated lesions with pronounced reddish-brown margins",
            "Tan, beige, or bleached necrotic centers inside the lesions",
            "Lesions drying out and tearing holes through the sword-like blades",
            "Yellow halos surrounding active lesion boundaries"
        ],
        "causes": [
            "Water settling in leaf folds or overhead splashing",
            "Warm, humid indoor microclimates with stagnant air",
            "Mechanical punctures from handling allowing fungal entrance"
        ],
        "immediate_actions": [
            "Quarantine the snake plant away from other houseplants.",
            "Prune out the affected blade with a sterile razor or shears, cutting below the lesion into healthy green tissue.",
            "Keep the entire plant completely dry."
        ],
        "treatment": {
            "organic": [
                "Dust clean cuts with ground cinnamon or bio-fungicide containing Bacillus subtilis.",
                "Apply cold-pressed neem oil to uninfected leaf areas to establish a protective barrier."
            ],
            "chemical": [
                "Spray affected and neighboring foliage with a liquid copper fungicide or chlorothalonil spray.",
                "Ensure spray dries quickly by placing a gentle circulating fan nearby."
            ]
        },
        "prevention": [
            "Water exclusively around the base at the soil rim; never pour water into the rosette or over blades.",
            "Water only once every 3 to 4 weeks—snake plants prefer extended dry periods.",
            "Ensure the soil mix has high porosity (cactus mix + perlite)."
        ]
    },

    "Snake_Plant_Healthy": {
        "class_id": "Snake_Plant_Healthy",
        "plant_name": "Snake Plant (Sansevieria)",
        "scientific_name": "Dracaena trifasciata",
        "condition_name": "Healthy Baseline",
        "is_healthy": True,
        "severity": "None",
        "category": "Healthy",
        "summary": "Stately, architectural upright sword foliage with firm leathery texture, crisp variegation, and zero fungal decay.",
        "symptoms": [
            "Rigid, stiff, erect sword-shaped blades with crisp margins",
            "Distinct banded marbling of deep green and golden-yellow borders",
            "Firm root crown with no mushiness or discoloration"
        ],
        "causes": [
            "Proper low-to-bright indirect lighting, fast-draining potting substrate, and disciplined underwatering."
        ],
        "immediate_actions": [
            "No action needed. Your Snake Plant is in excellent health!"
        ],
        "treatment": {
            "organic": [
                "Dust upright leaves periodically with a dry cloth."
            ],
            "chemical": [
                "No chemicals required."
            ]
        },
        "prevention": [
            "Water only when soil is bone dry all the way to the bottom (every 3-4 weeks in summer, 6-8 weeks in winter).",
            "Thrives in almost any indoor light from low light to bright indirect sunshine.",
            "Keep temperatures above 12°C to prevent cold shock."
        ]
    },

    "Snake_Plant_Leaf_Withering": {
        "class_id": "Snake_Plant_Leaf_Withering",
        "plant_name": "Snake Plant (Sansevieria)",
        "scientific_name": "Dracaena trifasciata",
        "condition_name": "Leaf Withering & Desiccation",
        "is_healthy": False,
        "severity": "Moderate",
        "category": "Physiological / Root Stress",
        "summary": "Loss of leaf turgor and cellular collapse caused by severe root disruption—either chronic dehydration or root rot preventing water intake.",
        "symptoms": [
            "Blades losing firmness, puckering, wrinkling, and developing vertical creases",
            "Leaves becoming limp, drooping, leaning, or bending under their own weight",
            "Dull, faded leaf luster and paper-thin blade cross-section"
        ],
        "causes": [
            "Primary Cause A (Underwatering): Soil left bone dry for months, causing cellular desiccation.",
            "Primary Cause B (Root Rot): Chronic overwatering rotted root hairs, preventing the plant from drinking water even in soggy soil!"
        ],
        "immediate_actions": [
            "DIAGNOSTIC TEST: Check root ball moisture immediately!",
            "If soil is dust-dry and roots are firm/white: Severe dehydration.",
            "If soil is damp/mushy and roots are brown, slimy, or foul: Root Rot."
        ],
        "treatment": {
            "organic": [
                "For Dehydration: Bottom-water by setting pot in a basin of lukewarm water for 45 minutes until soil is re-hydrated, then drain thoroughly.",
                "For Root Rot: Unpot, wash away soggy soil, trim all rotten brown roots with sterilized shears, dust with cinnamon/charcoal, and repot in fresh dry succulent mix."
            ],
            "chemical": [
                "If replanting after root rot, drench with mild potassium phosphite or copper fungicide to suppress lingering Pythium/Phytophthora root pathogens."
            ]
        },
        "prevention": [
            "Establish a reliable dry-then-drench watering routine.",
            "Never leave a snake plant standing in water filled decorative cachepots.",
            "Use heavy terracotta pots that breathe and allow excess moisture to evaporate."
        ]
    },

    "Spider_Plant_Fungal_leaf_spot": {
        "class_id": "Spider_Plant_Fungal_leaf_spot",
        "plant_name": "Spider Plant",
        "scientific_name": "Chlorophytum comosum",
        "condition_name": "Fungal Leaf Spot Disease",
        "is_healthy": False,
        "severity": "Moderate",
        "category": "Fungal Infection",
        "summary": "Fungal spotting caused by Alternaria, Fusarium, or Rhizoctonia creating scattered lesions along delicate arching spider plant ribbons.",
        "symptoms": [
            "Small circular or irregular water-soaked spots turning dark brown or reddish-black",
            "Concentric rings or yellow chlorotic borders surrounding leaf lesions",
            "Spots merging, causing entire leaf blades to collapse and turn necrotic"
        ],
        "causes": [
            "Overhead watering wetting the dense crown of leaves",
            "High relative humidity combined with poor indoor air circulation",
            "Overcrowded foliage retaining moisture pockets"
        ],
        "immediate_actions": [
            "Isolate the spider plant to protect neighboring plants.",
            "Trim off heavily infected ribbon leaves at the base using sterilized scissors.",
            "Clear away fallen dead debris from the soil surface."
        ],
        "treatment": {
            "organic": [
                "Spray with organic bio-fungicide containing Bacillus amyloliquefaciens.",
                "Apply a light neem oil spray (diluted 0.5%) in the evening out of direct sunlight."
            ],
            "chemical": [
                "Spray foliage with liquid copper octanoate or broad-spectrum chlorothalonil fungicide.",
                "Repeat every 10 days until clean, unspotted new growth emerges from the center."
            ]
        },
        "prevention": [
            "Water only at the soil level; keep water out of the center crown.",
            "Provide bright indirect light and increase room air circulation with a fan.",
            "Thin out overcrowded crowns by propagating or potting up spiderettes."
        ]
    },

    "Spider_Plant_Healthy": {
        "class_id": "Spider_Plant_Healthy",
        "plant_name": "Spider Plant",
        "scientific_name": "Chlorophytum comosum",
        "condition_name": "Healthy Baseline",
        "is_healthy": True,
        "severity": "None",
        "category": "Healthy",
        "summary": "Cascading vibrant arching leaves with crisp green and white variegation, healthy tubers, and active plantlet stolons.",
        "symptoms": [
            "Lush, arching strap-like leaves with vivid emerald and cream/white variegation",
            "Clean leaf tips without dark necrosis or yellowing halos",
            "Prolific production of long stolons bearing small spiderettes and white starry flowers"
        ],
        "causes": [
            "Ideal filtered light, moderate watering, well-draining potting soil, and non-fluoridated water."
        ],
        "immediate_actions": [
            "Your Spider Plant is flourishing! No remedial measures required."
        ],
        "treatment": {
            "organic": [
                "Snip mature plantlets ('spiderettes') and root them in water or potting soil to grow new plants!"
            ],
            "chemical": [
                "No chemicals required."
            ]
        },
        "prevention": [
            "Position in bright, indirect light (east or north-facing windows).",
            "Water when the top 1 inch of soil is dry; allow good pot drainage.",
            "Feed with diluted organic fertilizer once a month in spring and summer."
        ]
    },

    "Spider_Plant_Leaf_Tip_Necrosis": {
        "class_id": "Spider_Plant_Leaf_Tip_Necrosis",
        "plant_name": "Spider Plant",
        "scientific_name": "Chlorophytum comosum",
        "condition_name": "Leaf Tip Necrosis / Tip Burn",
        "is_healthy": False,
        "severity": "Mild to Moderate",
        "category": "Physiological / Abiotic Disorder",
        "summary": "Drying and browning of leaf tips caused by fluoride/boron toxicity from tap water, accumulated fertilizer salts, or low humidity.",
        "symptoms": [
            "Crisp, brown, blackened, or papery dry tips on the pointed ends of leaves",
            "Narrow yellow chlorotic band between dead brown tip and healthy green tissue",
            "Gradual progression from extreme tip down the leaf blade if irritants persist"
        ],
        "causes": [
            "Fluoride, chlorine, chloramines, or soluble salts present in municipal tap water",
            "Accumulation of fertilizer mineral salts in the potting mix",
            "Very dry indoor winter air (humidity below 35%) or erratic watering intervals"
        ],
        "immediate_actions": [
            "Switch watering source immediately: use rainwater, distilled water, or reverse osmosis water (spider plants are exceptionally sensitive to fluoride).",
            "Trim off dead brown tips with sharp clean scissors: cut at a slight angle following the natural leaf point, leaving a millimeter of brown so you do not wound living green tissue."
        ],
        "treatment": {
            "organic": [
                "Flush the pot thoroughly with 2-3 liters of distilled water to leach accumulated salts and minerals out of the drainage holes.",
                "Increase localized humidity by grouping plants together or running an ultrasonic cool mist humidifier (target 50-60% RH)."
            ],
            "chemical": [
                "Avoid superphosphate and perlite containing high fluoride levels.",
                "Halt chemical fertilizers for 4-6 weeks, then resume at only 1/4 label strength."
            ]
        },
        "prevention": [
            "Never use municipal fluoridated tap water on spider plants.",
            "Water consistently when the top 1 inch feels slightly dry—avoid letting soil swing from soggy to desert dry.",
            "Repot every 1-2 years to prevent tubered roots from compacting and trapping salts."
        ]
    }
}

def get_disease_info(class_id: str) -> dict:
    """Retrieve full care and treatment details for a given class ID."""
    return DISEASE_KNOWLEDGE_BASE.get(class_id, {
        "class_id": class_id,
        "plant_name": class_id.split("_")[0],
        "condition_name": class_id.replace("_", " "),
        "is_healthy": "healthy" in class_id.lower(),
        "severity": "Unknown",
        "category": "General",
        "summary": "Diagnostic profile information currently being updated.",
        "symptoms": ["Inspection required"],
        "causes": ["Environmental or pathogen factors"],
        "immediate_actions": ["Isolate plant and inspect under good lighting."],
        "treatment": {
            "organic": ["Apply neem oil spray and maintain proper soil moisture."],
            "chemical": ["Consult local agricultural specialist for targeted treatment."]
        },
        "prevention": ["Maintain balanced watering, bright indirect light, and good airflow."]
    })

def export_knowledge_base_json(output_path: Path):
    """Exports the complete disease knowledge base to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DISEASE_KNOWLEDGE_BASE, f, indent=2)
    print(f"Exported disease knowledge base to {output_path}")

if __name__ == "__main__":
    from src import config
    export_knowledge_base_json(config.DISEASE_KB_PATH)

