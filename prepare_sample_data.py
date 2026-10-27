"""
prepare_sample_data.py
Creates realistic sample technical manuals and troubleshooting guides for home appliances:
1. Washing Machine Guide (.txt)
2. Refrigerator Maintenance Manual (.txt)
3. Air Conditioner Troubleshooting Guide (.txt)
4. Microwave Oven Knowledge Base (.csv)
5. Dishwasher User Manual (.pdf)
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_manuals")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 1. Washing Machine Guide (TXT)
WASHING_MACHINE_TEXT = """# HOMEGUARD PRO AUTOMATIC WASHING MACHINE - USER & TROUBLESHOOTING MANUAL
Model Series: WM-8000 Front-Load & Top-Load
Category: Washing Machine

## 1. Safety Warnings & Electrical Hazard Guidelines
- DANGER: Always unplug the washing machine from the mains electrical outlet before performing any inspection, maintenance, or cleaning.
- Never touch electrical plugs or switches with wet hands.
- Do not bypass the child safety door lock mechanism.
- If the supply cord or power plug is damaged, it must be replaced by the manufacturer, an authorized service technician, or a certified electrician to avoid electrical fire or shock.
- Never wash items soaked in gasoline, paint thinner, dry-cleaning solvents, or other flammable liquids.

## 2. Common Error Codes and Troubleshooting

### Error Code E1: Water Inlet Failure (No Water Entering Drum)
- Possible Causes:
  1. Water supply tap is turned off or water pressure is too low.
  2. Inlet hose is kinked or twisted behind the machine.
  3. Inlet valve mesh filter is clogged with sediment, rust, or mineral deposits.
- Recommended Solutions:
  1. Verify the household water tap is fully open and water pressure is adequate.
  2. Inspect the water inlet hose and straighten any kinks.
  3. Turn off tap, unscrew inlet hose from the back of the washer, remove the plastic mesh filter with pliers, gently scrub under running water with a soft brush, and reinstall securely.

### Error Code E2: Drain Problem (Water Not Draining Out)
- Possible Causes:
  1. Drain hose is bent, crushed, or positioned higher than 100 cm above floor level.
  2. Debris filter / pump filter is obstructed by coins, buttons, hairpins, or lint.
  3. Drain pump impeller is jammed.
- Recommended Solutions:
  1. Check drain hose routing; ensure no severe bends and end of hose is not submerged in water.
  2. Place a shallow tray and towel under the bottom-front service access flap.
  3. Unscrew the drain pump filter cap counter-clockwise, let residual water drain, remove trapped debris, check that the impeller wheel spins freely with a finger, and tightly recap.
- Precaution: Water inside drum may be hot if cycle just completed; allow 30 minutes to cool before opening drain filter.

### Error Code E3 / dE: Door Lock / Latch Error
- Possible Causes:
  1. Door is not closed completely or laundry is caught between door glass and rubber gasket.
  2. Door strike pin is misaligned or switch mechanism is faulty.
- Recommended Solutions:
  1. Press door firmly until an audible click is heard before pressing Start/Pause.
  2. Check for trapped clothing around the door seal.
  3. If latch is stuck, do not force door open. Turn off power for 5 minutes to release thermal lock.

### Error Code UE: Unbalanced Load Error (Drum Cannot Spin)
- Possible Causes:
  1. A single heavy item (such as a bedsheet, bathmat, or heavy towel) is balled up on one side.
  2. The washing machine is not standing level on all four adjustable feet.
- Recommended Solutions:
  1. Pause cycle, open door, redistribute items evenly inside drum. Add 1-2 smaller items to balance heavy single loads.
  2. Use a spirit level on top of the machine and adjust the front leveling feet by turning locknuts clockwise/counter-clockwise until stable and firm.

## 3. Routine Care & Preventive Maintenance
- Drum Tub Clean: Run a monthly maintenance cycle at 60°C or 90°C with 100g of citric acid or tub descaler.
- Detergent Drawer Cleaning: Press release latch to remove the detergent drawer. Rinse under warm water to dissolve dried powder and fabric softener build-up.
- Rubber Bellow Gasket: Wipe dry after every wash to prevent mildew and odor formation. Leave door slightly ajar when not in use.
"""

# 2. Refrigerator Maintenance Manual (TXT)
REFRIGERATOR_TEXT = """# FROSTFREE PLUS SMART REFRIGERATOR - USER & SERVICE GUIDE
Model Series: RF-550 Multi-Door & Double-Door Frost Free
Category: Refrigerator

## 1. Critical Safety Cautions
- DANGER: Contains R600a / R134a refrigerant under pressure. Never attempt to pierce, solder, or cut refrigerant tubing. Sealed refrigeration circuit repairs must ONLY be executed by EPA-certified HVAC/R technicians.
- Unplug the unit before replacing LED light modules or cleaning behind the refrigerator.
- Do not use mechanical devices, sharp knives, or ice picks to accelerate defrosting; you risk puncturing the evaporator coils.
- Never store flammable gases, aerosols, or alcohol solvents inside the appliance.

## 2. Common Issues & Step-by-Step Remedies

### Issue: Refrigerator Is Not Cooling (Compressor Does Not Run or Runs Continuously)
- Possible Causes:
  1. Temperature setting is set too warm or Eco-mode is inadvertently active.
  2. Condenser coils on bottom/rear are covered in heavy lint, pet hair, and household dust.
  3. Inadequate clearance around the refrigerator preventing heat dissipation.
  4. Defrost thermostat or compressor relay failure.
- Recommended Solutions:
  1. Set refrigerator temperature to 3°C (37°F) and freezer to -18°C (0°F).
  2. Disconnect mains plug. Pull refrigerator gently forward. Vacuum and brush condenser coils using a soft coil brush every 6 months.
  3. Ensure at least 5 cm (2 inches) clearance at rear and sides, and 10 cm on top.
  4. If compressor clicks continuously every 2-3 minutes without starting, call an authorized technician to test the PTC start relay and run capacitor.

### Issue: Water Pooling at the Bottom of Fresh Food Compartment or Under Crisper Drawers
- Possible Causes:
  1. Defrost drain hole at the rear internal wall is blocked with ice, food particles, or algae.
  2. Drain tube behind refrigerator has kinked or frozen shut.
- Recommended Solutions:
  1. Remove crisper drawers. Locate the drain trough and drain funnel at the back wall.
  2. Flush the drain hole with warm water (not boiling) using a turkey baster or syringe.
  3. Gently pass a flexible nylon tube or pipe cleaner through the drain hole to clear organic blockage.
  4. Verify that water drains freely into the evaporation pan above compressor at the back.

### Issue: Excessive Frost Buildup on Freezer Walls and Ceiling
- Possible Causes:
  1. Magnetic door gasket is torn, dirty, hardened, or not sealing airtight.
  2. Door is being opened too frequently or held open for prolonged periods in high humidity.
  3. Defrost heater or bi-metal defrost thermostat has burnt out.
- Recommended Solutions:
  1. Test seal with the 'Dollar Bill Test': Close door on a dollar bill/paper slip. If it slides out with zero resistance, clean or replace gasket.
  2. Clean gasket grooves with mild soap and warm water; soften stiff spots using a warm damp cloth.
  3. If frost resembles hard snow packed over entire rear panel, the automatic defrost system is defective and requires technician inspection.

### Issue: Buzzing, Rattling, or Humming Noises
- Normal sounds: Mild humming is the inverter compressor; occasional crackling is thermal expansion of panels during defrost.
- Abnormal sounds: Vibrating metal tube at rear, unlevel feet, or fan blade hitting frost accumulation. Ensure leveling legs touch the floor firmly.
"""

# 3. Air Conditioner Troubleshooting Guide (TXT)
AIR_CONDITIONER_TEXT = """# COOLWAVE INVERTER SPLIT AIR CONDITIONER - TROUBLESHOOTING GUIDE
Model Series: AC-IN18 1.5 Ton / 2.0 Ton Eco Inverter
Category: Air Conditioner

## 1. Safety & Hazardous Voltage Warning
- HIGH VOLTAGE WARNING: The outdoor condenser unit contains high-voltage electronics (230V-400V) and large capacitors that hold lethal electrical charge even after disconnecting power.
- NEVER open outdoor unit casing or service the fan motor / inverter PCB yourself.
- Disconnect main breaker/isolator switch before cleaning the indoor unit filters.
- If you suspect a refrigerant leak (sweet chemical odor, hissing sound, or ice on thin suction copper pipe), immediately switch off the AC, ventilate the room, and contact certified technician.

## 2. Common Problems & Diagnostic Procedures

### Problem: AC Is Running But Not Cooling the Room (Blowing Warm or Room-Temperature Air)
- Possible Causes:
  1. Indoor air filters are heavily clogged with dust, choking airflow across the cooling coil.
  2. Mode setting is wrongly set to 'FAN' or 'DRY' instead of 'COOL' mode.
  3. Set temperature is higher than or equal to current room ambient temperature.
  4. Outdoor condenser coil is blocked by leaves, dust, or direct extreme sunlight.
  5. Low refrigerant charge due to copper flare joint micro-leak.
- Recommended Solutions:
  1. Open indoor unit front panel, slide out mesh air filters, wash with lukewarm water, allow to dry completely in shade, and reinstall.
  2. Ensure remote is in 'COOL' mode (snowflake icon) and set temperature between 22°C to 24°C.
  3. Inspect outdoor unit; maintain at least 30 cm clear airspace around it.
  4. If indoor evaporator coil is freezing into a block of ice or thin copper pipe is frosted, turn off AC immediately and call a certified technician to detect leak and recharge refrigerant.

### Problem: Water Dripping or Overflowing from Indoor Unit
- Possible Causes:
  1. Condensate drain pipe is choked with dirt, algae, or fungal slime.
  2. Condensate drain hose has reverse slope (sagging) or exit tip is submerged in water bucket.
  3. Indoor unit is not installed horizontally level on mounting bracket.
- Recommended Solutions:
  1. Check drain hose outlet outside. Ensure it slopes downwards with gravity.
  2. Blow gently into drain pipe or use a wet/dry shop vacuum at outdoor drain pipe exit to suck out clogged sludge.
  3. Pour 200 ml of clean water slowly onto the indoor cooling coil fins to confirm smooth water flow through drain tray.

### Problem: Error Codes on Display
- Error Code E1 / F1: Indoor room temperature sensor or coil thermistor open/short circuit. Requires sensor replacement by technician.
- Error Code E4: Compressor overload or indoor fan motor abnormal RPM. Turn off circuit breaker for 10 minutes to reset. If E4 returns, contact service center.
- Error Code EC: Refrigerant leakage detection algorithm triggered. Do not run unit to prevent compressor motor burnout.

## 3. Preventive Maintenance Schedule
- Every 2 Weeks: Wash indoor mesh air filters.
- Every 3 Months: Clean outdoor condenser fins using a garden hose (low pressure) from top to bottom.
- Annually: Pre-summer professional wet service with chemical wash of cooling coils and blower wheel.
"""

# 4. Microwave Oven Knowledge Base (CSV)
MICROWAVE_DATA = [
    {
        "Appliance": "Microwave Oven",
        "Category": "Microwave Oven",
        "Issue": "Sparking or Arcing inside Cavity",
        "Error_Code": "None",
        "Possible_Cause": "Metal utensils, aluminum foil, gold-rimmed plates, or burnt food debris on waveguide cover.",
        "Troubleshooting_Steps": "1. Immediately press STOP and unplug microwave. 2. Remove any metallic containers, twist-ties, or foil. 3. Inspect the mica waveguide cover on right inner wall; if coated with grease or scorched black, wipe clean with damp cloth or replace mica sheet. 4. Never run microwave empty.",
        "Safety_Precaution": "DANGER: Arcing can burn a hole through the metal wall and ruin magnetron. Do not use metal racks unless supplied by manufacturer."
    },
    {
        "Appliance": "Microwave Oven",
        "Category": "Microwave Oven",
        "Issue": "Microwave Runs and Light Turns On But Does Not Heat Food",
        "Error_Code": "H01 / F02",
        "Possible_Cause": "Blown high-voltage diode, shorted high-voltage capacitor, or defective magnetron.",
        "Troubleshooting_Steps": "1. Check if microwave was accidentally put in 'Demo Mode' or 'Timer Mode' (shows countdown without heating). 2. Test with 200ml cup of water for 60 seconds on HIGH power. If water is cold, internal high-voltage components have failed. 3. Call authorized appliance repair.",
        "Safety_Precaution": "CRITICAL LETHAL HAZARD: High voltage capacitor stores over 2,000 to 4,000 Volts even when unplugged. NEVER remove outer steel cover. Service must strictly be performed by trained technicians."
    },
    {
        "Appliance": "Microwave Oven",
        "Category": "Microwave Oven",
        "Issue": "Glass Turntable Plate Not Rotating",
        "Error_Code": "None",
        "Possible_Cause": "Turntable roller ring misaligned, food crumb obstruction in track, or defective turntable coupler motor.",
        "Troubleshooting_Steps": "1. Remove glass tray and circular roller support ring. 2. Wash glass tray and rollers with warm soapy water. 3. Clean drive hub and track on the floor of cavity. 4. Verify center plastic drive coupler is seated properly onto motor shaft. 5. Check if 'Turntable Off' button is active.",
        "Safety_Precaution": "Handle glass tray carefully when hot; thermal shock can shatter glass if immersed in cold water while hot."
    },
    {
        "Appliance": "Microwave Oven",
        "Category": "Microwave Oven",
        "Issue": "Keypad / Touch Screen Display Not Responding",
        "Error_Code": "SE / 5E / Key Error",
        "Possible_Cause": "Child Lock engaged, moisture on touch membrane, or stuck button switch.",
        "Troubleshooting_Steps": "1. Check for 'Lock' or 'Padlock' icon on display. Press and hold 'STOP/CLEAR' or 'LOCK' button for 3 to 5 seconds until beep sounds to deactivate Child Lock. 2. Wipe keypad with dry microfiber cloth. 3. Unplug microwave for 5 minutes to reset microcontroller logic board.",
        "Safety_Precaution": "Do not spray liquid cleaners or water directly onto touch panel; use lightly dampened cloth."
    },
    {
        "Appliance": "Microwave Oven",
        "Category": "Microwave Oven",
        "Issue": "Burning Smell or Smoke While Operating",
        "Error_Code": "None",
        "Possible_Cause": "Severe food splatter inside cavity, scorched microwave popcorn bag, or overheating electrical component.",
        "Troubleshooting_Steps": "1. Press STOP and unplug immediately. 2. Keep door closed to starve smoke of oxygen if fire is suspected. 3. After cavity cools, thoroughly clean internal walls, ceiling, and floor with baking soda and water solution. 4. If electrical burning smell persists without any food inside, discontinue use.",
        "Safety_Precaution": "FIRE HAZARD: If flames occur inside cavity, keep door firmly shut, turn off mains power, and do not open door until fire is extinguished."
    }
]

# 5. Dishwasher User Manual (PDF)
def generate_dishwasher_pdf(filepath):
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=10
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6
    )
    
    alert_style = ParagraphStyle(
        'Alert',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#B91C1C'),
        spaceBefore=4,
        spaceAfter=8
    )

    story = []
    
    story.append(Paragraph("AQUACLEAN DELUXE DISHWASHER - TECHNICAL GUIDE & USER MANUAL", title_style))
    story.append(Paragraph("<b>Appliance:</b> Dishwasher | <b>Model Series:</b> DW-600 Built-In & Freestanding", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Critical Safety & Flood Prevention Guidelines", h2_style))
    story.append(Paragraph("Always shut off the water supply stopcock and isolate electrical power at the circuit breaker before attempting any internal filter or pump servicing.", body_style))
    story.append(Paragraph("<b>WARNING:</b> Internal heating element reaches high temperatures (65°C - 75°C). Never open door during drying cycle to avoid steam scalds.", alert_style))
    
    story.append(Paragraph("2. Common Faults and Solutions", h2_style))
    
    story.append(Paragraph("<b>Error Code E15: AquaStop Flood Protection Sensor Triggered</b>", body_style))
    story.append(Paragraph("<b>Possible Cause:</b> Water has leaked into the internal bottom drip tray beneath the machine, lifting the safety styrofoam float switch.", body_style))
    story.append(Paragraph("<b>Recommended Solution:</b> 1. Turn off water tap and unplug electrical power. 2. Gently tilt dishwasher backward by 45 degrees over a towel to drain trapped water from bottom base pan. 3. Check inlet water connection and door gasket for visible leaks. 4. If E15 recurs, call authorized technician.", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Error Code E24 / E25: Dishwasher Not Draining (Standing Water in Sump)</b>", body_style))
    story.append(Paragraph("<b>Possible Cause:</b> Triple fine filter in floor is clogged with food grease, or foreign object (glass shard, bone piece) jammed in drain pump impeller.", body_style))
    story.append(Paragraph("<b>Recommended Solution:</b> 1. Remove lower rack. 2. Turn central cylindrical filter counter-clockwise and pull out. 3. Wash mesh filter under hot running water with liquid soap. 4. Remove white plastic pump impeller cover; inspect impeller wheel with a flashlight and remove any debris with tweezers. 5. Click cover firmly back into place before running.", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Dishes Coming Out Dirty or With White Film / Water Spots</b>", body_style))
    story.append(Paragraph("<b>Possible Cause:</b> Clogged spray arm nozzles, incorrect detergent dosing, empty rinse aid dispenser, or low dishwasher salt in water softener unit.", body_style))
    story.append(Paragraph("<b>Recommended Solution:</b> 1. Unclip upper and lower spray arms; use a wooden toothpick to poke out mineral seeds from spray jet holes. 2. Refill rinse aid reservoir until indicator turns dark. 3. In hard water areas, fill the salt container with special coarse dishwasher regenerating salt.", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Monthly Maintenance & Filter Hygiene", h2_style))
    story.append(Paragraph("Run an empty Intensive cycle (70°C) with a specialized dishwasher cleaning tablet every 30 cycles to remove hidden grease and limescale deposits inside pipes.", body_style))
    
    doc.build(story)

def create_all_sample_files():
    # 1. TXT 1
    wm_path = os.path.join(SAMPLE_DIR, "washing_machine_guide.txt")
    with open(wm_path, "w", encoding="utf-8") as f:
        f.write(WASHING_MACHINE_TEXT.strip())
    print(f"Created: {wm_path}")

    # 2. TXT 2
    rf_path = os.path.join(SAMPLE_DIR, "refrigerator_manual.txt")
    with open(rf_path, "w", encoding="utf-8") as f:
        f.write(REFRIGERATOR_TEXT.strip())
    print(f"Created: {rf_path}")

    # 3. TXT 3
    ac_path = os.path.join(SAMPLE_DIR, "air_conditioner_guide.txt")
    with open(ac_path, "w", encoding="utf-8") as f:
        f.write(AIR_CONDITIONER_TEXT.strip())
    print(f"Created: {ac_path}")

    # 4. CSV
    csv_path = os.path.join(SAMPLE_DIR, "microwave_troubleshooting.csv")
    df = pd.DataFrame(MICROWAVE_DATA)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Created: {csv_path}")

    # 5. PDF
    pdf_path = os.path.join(SAMPLE_DIR, "dishwasher_user_guide.pdf")
    generate_dishwasher_pdf(pdf_path)
    print(f"Created: {pdf_path}")

if __name__ == "__main__":
    create_all_sample_files()
