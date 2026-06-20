# Review queue — golden v0.2.0 (eval3 increment)

31 new scenes · 205 gate cases · 48 label records total. The model proposed; you dispose. For each rising edge confirm/adjust the timestamp; for each gate case keep or flip the verdict. Tell me the corrections and I'll apply them and stamp the records reviewed.

## eval3 — Top Security Camera Fails 2026 (AFV compilation of many short security-cam clips)

_37 shot-segmented incidents from the compilation._

### eval3_s01  `0.0–6.0s`  — A child riding a mini-bike crashes into a fence and falls. An adult quickly runs to check on them and the bike.
- ⏱ **rising edge** @ `0.00s` · A person is riding a mini-bike  
  ↳ _Person_1 appears on screen riding a mini-bike._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `2.00s` · A person crashes a mini-bike  
  ↳ _The mini-bike hits the fence and the rider falls off._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `2.00s` · A person falls  
  ↳ _Person_1 is thrown from the mini-bike upon impact with the fence._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `2.50s` · An adult runs to assist  
  ↳ _Person_2 (adult male) runs quickly towards the crash site after the fall._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `3.00s` · Person_1 is getting up  
  ↳ _Person_1 is seen moving and getting up from the ground behind the fence._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0.0-6.0: A person riding a mini-bike crashes, and another person comes to assist. This is an accident, not suspicious activity.  
  ↳ notes: The event is clearly an accident and not indicative of any suspicious behavior.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 2.0: A person riding a mini-bike crashes into a fence and falls. An adult immediately runs to assist, indicating a potential need for help.  
  ↳ notes: A crash and fall from a motorized vehicle is a clear indicator of a potential emergency or need for help, even if injuries are not visible.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 2.0: A person crashes a mini-bike and falls. An adult immediately runs to them. While help is clearly needed and provided, the clip does not contain audio or visual cues of the person explicitly asking for help.  
  ↳ notes: The situation strongly implies a need for help, and help is provided, but there's no explicit 'want' expressed. A system might infer 'wanted' from 'needed'.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 2.0: Person_1, who appears to be a child, crashes a mini-bike into a fence and falls. This is a dangerous situation for a child.  
  ↳ notes: The rider is clearly a child or young teenager, and crashing a motorized vehicle is an inherently dangerous event.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0.0-6.0: No elder individuals are present in the video. The person riding the mini-bike is a child, and the assisting person is an adult.  
  ↳ notes: No elder is visible in the clip.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.0-6.0: All activity occurs in the backyard, far from any visible door.  
  ↳ notes: The camera view is of a backyard, not an entrance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s02  `34.7–38.2s`  — A person slips on icy steps while carrying a box, falling and scattering its contents.
- ⏱ **rising edge** @ `35.46s` · A person falls  
  ↳ _Person loses balance and falls down steps_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `34.66s` · Someone is at the door  
  ↳ _Person is visible at the door_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent (e.g., lurking, forced entry).  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0.8-3.5 - Person slips and falls hard on the steps, contents of box scatter. They remain on the ground for a moment.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 0.8-3.5 - Person slips and falls hard on the steps. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a young to middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.0-0.8 - A person is visible walking down the steps from the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s03  `38.2–41.7s`  — A person carrying two large Easter eggs slips on the grass and falls.
- ⏱ **rising edge** @ `39.66s` · A person falls  
  ↳ _Person loses balance and falls on the grass_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `38.16s` · Someone is at the door  
  ↳ _Person is visible approaching the door_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 5.0-7.0 - Person slips and falls on the grass, landing hard. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 5.0-7.0 - Person slips and falls on the grass. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 3.5-5.0 - A person is visible approaching the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s04  `41.7–45.2s`  — A person slips on a patio and falls onto the grass.
- ⏱ **rising edge** @ `42.66s` · A person falls  
  ↳ _Person loses balance and falls on the grass_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `41.66s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 8.0-10.5 - Person slips and falls on the grass, landing hard. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 8.0-10.5 - Person slips and falls on the grass. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 7.0-8.0 - A person is visible walking on the patio near the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s05  `45.2–48.2s`  — A person runs down steps, slips, and falls on the concrete.
- ⏱ **rising edge** @ `45.66s` · A person falls  
  ↳ _Person loses balance and falls on the concrete_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `45.16s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just running and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 11.0-13.5 - Person slips and falls hard on the concrete. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 11.0-13.5 - Person slips and falls on the concrete. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a young adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 10.5-11.0 - A person is visible running from the door area.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s06  `48.2–50.7s`  — A person carrying a box slips on snow/ice and falls.
- ⏱ **rising edge** @ `48.66s` · A person falls  
  ↳ _Person loses balance and falls on snow/ice_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `48.16s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 14.0-16.0 - Person slips and falls hard on the snow/ice. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 14.0-16.0 - Person slips and falls on the snow/ice. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a young to middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 13.5-14.0 - A person is visible walking towards the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s07  `50.7–53.7s`  — A person walks down steps, slips, and falls.
- ⏱ **rising edge** @ `51.66s` · A person falls  
  ↳ _Person loses balance and falls down steps_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `50.66s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 17.0-19.0 - Person slips and falls hard on the steps. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 17.0-19.0 - Person slips and falls on the steps. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 16.0-17.0 - A person is visible walking down the steps from the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s08  `53.7–56.7s`  — A person carrying a laptop slips on steps and falls.
- ⏱ **rising edge** @ `54.66s` · A person falls  
  ↳ _Person loses balance and falls down steps_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `53.66s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 20.0-22.0 - Person slips and falls hard on the steps. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 20.0-22.0 - Person slips and falls on the steps. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a young adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 19.0-20.0 - A person is visible walking down the steps from the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s09  `56.7–60.2s`  — A person carrying a large object slips on steps and falls.
- ⏱ **rising edge** @ `57.66s` · A person falls  
  ↳ _Person loses balance and falls down steps_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `56.66s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just walking and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 23.0-25.5 - Person slips and falls hard on the steps. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 23.0-25.5 - Person slips and falls on the steps. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a young to middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 22.0-23.0 - A person is visible walking down the steps from the door.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s10  `60.2–63.2s`  — A person runs down steps, slips, and falls.
- ⏱ **rising edge** @ `60.66s` · A person falls  
  ↳ _Person loses balance and falls down steps_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `60.16s` · Someone is at the door  
  ↳ _Person is visible at the door area_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ No actions indicating suspicious intent.  
  ↳ notes: The person is clearly just running and falls.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 26.0-28.5 - Person slips and falls hard on the steps. They remain on the ground.  
  ↳ notes: The fall looks painful, and the person remains on the ground, indicating potential injury or need for assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 26.0-28.5 - Person slips and falls on the steps. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a hard fall often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a young adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 25.5-26.0 - A person is visible running from the door area.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s11  `63.2–66.2s`  — A person on a ladder carrying a large wooden board falls as the ladder tips.
- ⏱ **rising edge** @ `64.16s` · A person falls  
  ↳ _Ladder tips and person falls_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `63.16s` · Someone is at the door  
  ↳ _Person is visible near the door area with a ladder_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ The person is engaged in what appears to be home maintenance or construction, not suspicious activity.  
  ↳ notes: The person is clearly working with a ladder and materials.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 29.5-31.5 - Person falls from a ladder while carrying a large board. They remain on the ground.  
  ↳ notes: Falling from a ladder, especially with an object, is a significant event and likely requires assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 29.5-31.5 - Person falls from a ladder. No explicit call for help, but the fall is significant.  
  ↳ notes: While no explicit call for help, a fall from a ladder often implies a need for assistance. It's better to ask than assume they are fine.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person appears to be a middle-aged adult.  
  ↳ notes: The person does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 28.5-29.5 - A person is visible on a ladder near the door area.  
  ↳ notes: The person is clearly present at the door area.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s12  `92.9–108.9s`  — A man is cleaning a backyard basketball court when an above-ground pool bursts, flooding the area. The man moves to safety.
- ⏱ **rising edge** @ `94.88s` · The above-ground pool bursts  
  ↳ _The side wall of the above-ground pool collapses, releasing a large volume of water._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `94.88s` · A large volume of water is released  
  ↳ _Water from the burst pool rapidly flows across the backyard._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `96.88s` · The basketball court is flooded  
  ↳ _Water from the burst pool covers the entire basketball court._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ The man is engaged in a normal activity (cleaning) when an accidental event (pool bursting) occurs. There is no indication of malicious intent or suspicious behavior.  
  ↳ notes: The event is clearly an accident, not suspicious activity.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ At 0:02, an above-ground pool bursts, releasing a large volume of water that floods the backyard. The man present is caught by the initial surge but quickly moves to the side (0:06-0:08) and does not appear injured, trapped, or in distress. While a property incident, it does not constitute a personal emergency requiring immediate external help for the individual.  
  ↳ notes: The pool bursting is a significant property event, but the individual is not in immediate physical danger and does not appear to require assistance. It's an accident, not a human emergency.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ The man does not signal for help, call out, or display any behavior indicating he wants assistance. He simply moves away from the water.  
  ↳ notes: No explicit or implicit request for help is observed.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person visible in the clip is an adult, not a child. Furthermore, the adult is not in danger.  
  ↳ notes: The individual is clearly an adult, and not in danger.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person visible in the clip appears to be a middle-aged adult, not an elder. Additionally, the person is not in danger.  
  ↳ notes: The individual is an adult, not an elder, and is not in danger.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ The camera view is of a backyard with a basketball court and an above-ground pool, not an entrance or door.  
  ↳ notes: The scene is a backyard, not a door or entrance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s13  `133.1–138.1s`  — A man walks towards a front door, stumbles, falls into bushes, and then slowly gets up, appearing to be in pain.
- ⏱ **rising edge** @ `133.05s` · Someone is at the door  
  ↳ _Man approaches the door_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `135.05s` · Someone needs help  
  ↳ _Man stumbles and falls, then struggles to get up._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0:00-0:05 - Man is walking normally, then stumbles and falls. No indication of malicious intent.  
  ↳ notes: The fall is accidental, not suspicious.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0:02-0:05 - Man stumbles, falls into the bushes, and then slowly and with apparent difficulty gets up.  
  ↳ notes: The fall and struggle to get up clearly indicate a potential need for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 0:02-0:05 - Man falls and struggles to get up. He doesn't explicitly ask for help, but his actions suggest he might need it.  
  ↳ notes: While he clearly needs help, he doesn't vocalize it. The system should ask to verify if help is wanted.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0:00-0:05 - The person is an adult male.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0:00-0:05 - The person appears to be a middle-aged adult, not an elder.  
  ↳ notes: The person is clearly not an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0:00-0:01 - A man walks up to the front door area.  
  ↳ notes: The man is clearly approaching the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s14  `138.1–138.8s`  — A woman walks towards a different front door, holding a gift.
- ⏱ **rising edge** @ `138.05s` · Someone is at the door  
  ↳ _Woman walks towards the door._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0:05-0:07 - Woman walks normally towards the door, holding a gift. No suspicious behavior.  
  ↳ notes: The woman's actions are normal and non-threatening.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0:05-0:07 - Woman is walking normally, appears fine.  
  ↳ notes: The woman shows no signs of distress or needing help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0:05-0:07 - Woman is walking normally, not indicating any need for help.  
  ↳ notes: The woman is not asking for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0:05-0:07 - The person is an adult female.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0:05-0:07 - The person appears to be a middle-aged adult, not an elder.  
  ↳ notes: The person is clearly not an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0:05-0:07 - A woman walks up to the front door area.  
  ↳ notes: The woman is clearly approaching the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s15  `166.5–172.5s`  — A woman struggles to move a clothing rack on a porch, causing it to tip over and spill clothes. She then tries to right the rack.
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0.0-6.0: A woman is moving a clothing rack on a porch. It tips over, and she struggles to right it. This is clumsy, not suspicious.  
  ↳ notes: The activity involves a struggle and an object falling, which might initially seem unusual, but it's clearly an accident, not suspicious behavior like theft or vandalism.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0.0-6.0: A woman struggles to move a clothing rack, which then tips over. She is able to hold onto it and appears to be recovering without injury or further immediate danger.  
  ↳ notes: While the woman is struggling and the rack falls, she doesn't appear to be injured or in an emergency. She is actively trying to fix the situation herself. No explicit distress signal.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0.0-6.0: The woman struggles with the clothing rack but does not call for help or show signs of wanting assistance. She is actively trying to manage it herself.  
  ↳ notes: The woman is clearly having difficulty, but she doesn't vocalize or gesture for help. The system should not infer a request for help from mere struggle.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0.0-6.0: No child is present in the video.  
  ↳ notes: The person struggling is an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0.0-6.0: The woman appears to be an adult, not an elder. She is also not in danger.  
  ↳ notes: The person struggling is an adult, not an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.0-6.0: A woman is already on the porch, moving a clothing rack. She is not approaching the door as a visitor would.  
  ↳ notes: The woman is already present on the porch and engaged in an activity, not arriving at the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s16  `231.1–236.1s`  — An adult male on a porch performs a dance-like movement, slips, falls off the porch, and then quickly gets up, laughing.
- ⏱ **rising edge** @ `231.15s` · Someone is at the door  
  ↳ _Man appears on porch_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0.0-0.05: Man is on the porch, slips, falls, and gets up. His actions are clumsy/comedic, not suspicious.  
  ↳ notes: The man's exaggerated movements and fall might initially seem unusual, but he quickly recovers and laughs, indicating it was an accident or prank, not suspicious activity.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0.02-0.03: Man falls off the porch. 0.03-0.04: He immediately gets up. 0.04-0.05: Laughter is heard, and he walks away seemingly uninjured.  
  ↳ notes: The fall could be mistaken for an emergency, but the quick recovery and laughter clearly indicate no serious injury or need for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0.04-0.05: Laughter is heard after the fall, and the man walks away without any signs of distress or calling for help.  
  ↳ notes: No verbal or non-verbal cues indicate help is wanted.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0.0-0.05: No child is present in the video.  
  ↳ notes: The only person present is an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0.0-0.05: The person is an adult male, not an elder, and is not in danger.  
  ↳ notes: The person appears to be a middle-aged adult, not an elder, and recovers quickly from the fall.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.0-0.05: An adult male is clearly visible on the porch, near the door, for the entire duration of the clip.  
  ↳ notes: The man is present on the porch, which is considered 'at the door'.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s17  `258.4–267.4s`  — A young man is on a porch, then runs to a parked black SUV, opens its door, and then runs back to the house.
- ⏱ **rising edge** @ `258.38s` · Someone is at the door  
  ↳ _A person is visible on the porch at the start of the clip._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ A young man runs out of the house, gets into a car, and runs back inside. His actions appear hurried but not indicative of danger or crime.  
  ↳ notes: The man's quick movements might initially seem suspicious, but he appears to be a resident, entering and exiting his own property and vehicle. No forced entry, no signs of distress or theft.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ The man runs quickly, but there are no signs of injury, distress, or a dangerous situation. He gets into and out of a car and re-enters the house.  
  ↳ notes: His hurried actions could be misinterpreted as an emergency, but he seems to be in control and not in danger. He might have forgotten something or be in a rush.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ No verbal or non-verbal cues indicating a request for help.  
  ↳ notes: The man does not call for help or show any signs of needing assistance.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The person is an adult, not a child. No danger is observed.  
  ↳ notes: The person is clearly a young adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person is an adult, not an elder. No danger is observed.  
  ↳ notes: The person is clearly a young adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ A person is on the porch at the beginning of the clip (0.0s) and returns to the door area at the end (8.0s).  
  ↳ notes: The person is clearly visible at the door area at multiple points.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s18  `306.0–326.0s`  — A woman attempts to load two small dogs into the back of her SUV. The dogs are very playful and keep jumping out or resisting, causing the woman to become frustrated.
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0.00-20.00: A woman is seen trying to put two dogs into a car. The dogs are playful and resist. Her actions are consistent with pet ownership, not suspicious behavior.  
  ↳ notes: The activity might initially look like a struggle, but it's clearly a person dealing with pets, not a crime or suspicious act.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0.00-20.00: The woman is frustrated and struggles to get the dogs into the car, but she is not in any physical danger or distress that would require emergency help. Her vocalizations are of exasperation, not pain or fear.  
  ↳ notes: The woman's vocalizations and body language indicate frustration with her pets, not a need for emergency assistance. This is a common scenario for pet owners.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0.00-20.00: The woman expresses frustration verbally ('What are they doing?'), but she does not explicitly ask for help from an external party. Her actions are focused on managing her dogs herself.  
  ↳ notes: While she is clearly frustrated, she is not calling for help from others. The context is a common pet owner struggle.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0.00-20.00: No children are visible in the video.  
  ↳ notes: No children are present.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0.00-20.00: The woman appears to be a middle-aged adult, not an elder. No elders are visible in the video.  
  ↳ notes: No elders are present.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.00-20.00: The woman is in the driveway next to a parked car, not near any door of a building.  
  ↳ notes: The activity is in the driveway, not at a door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s19  `361.5–366.5s`  — A dog approaches the front porch, jumps onto it, and in its excitement, knocks over a large potted plant before running away.
- ⏱ **rising edge** @ `361.49s` · A dog is present on the porch.  
  ↳ _The dog runs into the frame and onto the porch._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `362.49s` · A potted plant is knocked over.  
  ↳ _The dog bumps into the plant, causing it to fall over with a crash._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0:00-0:03 - A dog runs onto the porch and knocks over a plant. This is mischief, not suspicious activity.  
  ↳ notes: The activity is caused by a pet, not a human, and does not indicate malicious intent or suspicious behavior.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0:00-0:05 - A dog knocks over a plant. No person or animal appears to be in distress or danger.  
  ↳ notes: No emergency or need for help is evident. The dog appears unharmed and playful.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0:00-0:05 - No person is visible, and no sounds or actions indicate that help is wanted.  
  ↳ notes: There is no human presence or any indication of a request for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0:00-0:05 - No child is present in the video.  
  ↳ notes: No child is observed in the scene.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0:00-0:05 - No elder is present in the video.  
  ↳ notes: No elder is observed in the scene.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0:00-0:03 - A dog is on the porch near the door. 'Someone' implies a human, not an animal.  
  ↳ notes: While an entity (a dog) is at the door, the goal condition 'someone' typically refers to a human. Therefore, this should not fire for an animal.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s20  `405.8–411.8s`  — An adult runs into a backyard, trips over something, falls hard onto the grass, and remains on the ground while two dogs approach them.
- ⏱ **rising edge** @ `406.68s` · A person is present  
  ↳ _An adult runs into the frame from the left._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `407.68s` · A person falls  
  ↳ _The adult trips and falls hard onto the grass._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0.0-6.0: An adult accidentally trips and falls in their own backyard. This is not suspicious activity.  
  ↳ notes: The event is an accidental fall, not indicative of any suspicious behavior or malicious intent.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 1.9-6.0: An adult falls hard onto the ground and remains unmoving for the remainder of the clip. The impact appears significant, and the lack of immediate movement suggests potential injury or being stunned.  
  ↳ notes: The person falls with considerable force and does not get up immediately. It is unclear from the clip if they are injured, momentarily stunned, or simply recovering. Further information or observation would be needed to confirm if help is required.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0.0-6.0: The person does not vocalize or signal for help after falling. The sounds heard are of the fall and dog barking.  
  ↳ notes: No explicit request for help is observed or heard in the clip.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0.0-6.0: The person who falls is an adult. No children are present in the video.  
  ↳ notes: No child is present in the scene.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0.0-6.0: The person who falls appears to be an adult, but there is no clear visual evidence to confirm they are an elder. The primary concern is the fall itself, not specifically an elder in danger.  
  ↳ notes: The person's age cannot be definitively determined as an elder from the video. The general 'emergency' condition covers the fall.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.0-6.0: The event occurs in a backyard, not at a door. The person is inside the property.  
  ↳ notes: The location of the incident is a backyard, not a door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s21  `476.8–482.8s`  — An adult woman and a child girl walk up the steps to a front door. The woman is carrying multiple bags and a backpack, while the child carries a pink item and exclaims "Oh my gosh."
- ⏱ **rising edge** @ `476.77s` · Someone is at the door  
  ↳ _Two people are seen walking up the steps towards the door._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ Two people, an adult and a child, are calmly walking up to the door. There are no signs of forced entry, unusual behavior, or loitering.  
  ↳ notes: The presence of people at the door is normal. No suspicious context.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ The adult and child are walking normally. The child says 'Oh my gosh' in a playful or surprised tone, not distressed. No visible injuries or signs of distress.  
  ↳ notes: The child's exclamation 'Oh my gosh' might be misinterpreted as distress, but the tone and context (walking calmly with an adult) suggest otherwise.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ No one is speaking in a way that indicates they need help. The child's 'Oh my gosh' is not a call for help.  
  ↳ notes: The child's exclamation is not a request for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ The child is walking with an adult, appears calm, and is not in any visible danger or distress. The adult is also calm.  
  ↳ notes: The child is accompanied by an adult, and there are no signs of danger.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ There is no elder visible in the clip.  
  ↳ notes: No elder present in the scene.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ At 0.0s, an adult and a child are seen walking up the steps towards the front door. They remain in view until the end of the clip at 6.0s.  
  ↳ notes: Clear visual evidence of two individuals approaching and present at the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s22  `505.0–512.0s`  — An adult is cleaning a pool with a long pole when a dog runs up and pushes them into the water. The adult quickly resurfaces, and the dogs run around the pool.
- ⏱ **rising edge** @ `508.05s` · A person falls  
  ↳ _The adult is pushed by a dog and begins to fall into the pool._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `508.55s` · A person is in the pool  
  ↳ _The adult fully enters the pool after being pushed._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `508.05s` · A pet interacts with a person  
  ↳ _A dog jumps on the adult, causing them to fall._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0-7: A dog playfully or accidentally pushes an adult into a pool. There is no indication of malicious intent or suspicious behavior.  
  ↳ notes: The interaction between the dog and the adult appears to be an accident or playful, not suspicious.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 3-7: An adult falls into a pool but quickly resurfaces and shows no signs of distress or struggle. The fall appears to be comedic rather than an emergency.  
  ↳ notes: Despite falling into the pool, the adult shows no signs of needing help and quickly recovers.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 3-7: The adult falls into the pool but does not call for help, wave, or exhibit any behavior indicating they want help.  
  ↳ notes: No verbal or non-verbal cues suggest that help is wanted.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0-7: The person in the video is an adult, not a child. No children are visible in the scene.  
  ↳ notes: No child is present in the video.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0-7: The person in the video is an adult, not an elder. No elders are visible in the scene.  
  ↳ notes: No elder is present in the video.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0-7: The video shows a backyard with a pool. There is no door visible, nor any indication of someone being at a door.  
  ↳ notes: The scene is a backyard, not an entrance with a door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s23  `541.5–553.5s`  — A man attempts to dunk a basketball, misses, and falls to the ground, then quickly recovers. A child watches from the porch.
- ⏱ **rising edge** @ `544.40s` · A person is falling  
  ↳ _Man jumps and begins to fall to the ground after missing the dunk._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `545.40s` · A person is on the ground  
  ↳ _Man lands on the ground after falling._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `541.50s` · A child is present  
  ↳ _Child visible on the porch from the start of the clip._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `541.50s` · Someone is playing basketball  
  ↳ _Man is holding a basketball and a basketball hoop is visible._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _suspicious activity_  
  ↳ The man is playing basketball and falls, which is not suspicious activity.  
  ↳ notes: The fall might look alarming but it's part of a game and not indicative of suspicious behavior.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _emergency or someone needs help_  
  ↳ Man falls at 0:03, but quickly gets up by 0:06 and continues playing. Audio includes a sound of surprise/pain followed by a laugh, and a woman's voice asking 'What in the world?' without alarm.  
  ↳ notes: The fall is dramatic, but the quick recovery and audio context (laughing, non-alarmed voice) indicate it was not a serious injury or emergency.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _help is wanted_  
  ↳ No verbal or non-verbal cues indicating a request for help from any individual.  
  ↳ notes: The man falls but does not call for help. The woman's voice is curious, not a call for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _child in danger_  
  ↳ The child is standing on the porch, a safe distance from the man's fall, and appears unharmed and calm.  
  ↳ notes: The child is not involved in the man's fall and is in a safe location.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _elder in danger_  
  ↳ No elder individuals are visible in the clip. The man is an adult, not an elder.  
  ↳ notes: The individuals present are an adult and a child.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _someone at the door_  
  ↳ The man is in the driveway, and the child is on the porch, neither are at the door.  
  ↳ notes: The camera view includes the porch and driveway, but no one approaches the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s24  `598.5–603.8s`  — A woman in a red dress is taking out a large white trash bag to a bin. As she lifts it, the bag rips, and all the trash falls onto the ground. She then looks down at the spilled contents.
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0.0-5.3: A woman is taking out trash, which accidentally spills. This is a common household mishap, not suspicious.  
  ↳ notes: The event is a mundane accident, not indicative of any criminal or unusual activity.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0.0-5.3: A woman accidentally spills trash. While inconvenient, this is not an emergency or a situation where she needs immediate help.  
  ↳ notes: The situation is a minor inconvenience, not a safety or health emergency.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0.0-5.3: The woman reacts to the spilled trash but does not call for help, show distress, or make any gestures indicating a need for assistance.  
  ↳ notes: The woman appears to be handling the situation herself, albeit with a minor setback.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0.0-5.3: No child is visible in the frame at any point.  
  ↳ notes: The only person present is an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0.0-5.3: The woman appears to be an adult, not an elder. She moves freely and without difficulty, even after the spill.  
  ↳ notes: The person is an adult, not an elder, and is not in danger.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0.0-5.3: The woman is in a driveway/yard area, next to trash bins, not at a door.  
  ↳ notes: The camera view is of an outdoor area, not an entrance door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s25  `630.7–636.7s`  — A person on a ladder working on a house falls when the ladder collapses. A man and child in the foreground witness the event.
- ⏱ **rising edge** @ `632.21s` · A person is in danger  
  ↳ _The ladder collapses and a person falls from a significant height._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `632.71s` · Someone needs help  
  ↳ _A person falls from a significant height, indicating a potential injury and need for assistance._  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ The activity involves a person working on a ladder, which then collapses. This is an accident, not suspicious activity.  
  ↳ notes: The event is an accident, not criminal or suspicious behavior.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ At 0:01.5, a ladder collapses, and a person falls from a significant height near the house. The person lands out of frame but the fall is clear.  
  ↳ notes: The fall is clearly visible and indicates a potential injury, requiring immediate attention.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ At 0:01.5, a person falls from a ladder. While no explicit call for help is heard, a fall from this height strongly implies a need for assistance.  
  ↳ notes: A fall from a ladder is an emergency situation where help is implicitly wanted, even if not explicitly requested verbally.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ A child is present in the foreground but is not directly involved in the ladder fall and appears safe.  
  ↳ notes: The child is a witness to the accident, not a victim or in immediate danger.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ The person who falls from the ladder appears to be an adult, not an elder.  
  ↳ notes: The person involved in the accident does not appear to be an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ No one is seen approaching or at the door of the house.  
  ↳ notes: The activity is in the driveway and near the side of the house, not at the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s26  `676.6–690.6s`  — A man attempts to stop a rolling ATV with a trailer, gets dragged down a hill, and then gets up unharmed.
- ⏱ **rising edge** @ `679.59s` · A vehicle is moving uncontrollably  
  ↳ _ATV starts rolling downhill unexpectedly_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `680.59s` · A person is in danger  
  ↳ _Man falls and is dragged by the ATV_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `680.59s` · A person needs help  
  ↳ _Man falls and is dragged by the ATV, indicating potential injury or distress_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0:00-0:14 - A man is trying to control his own ATV and trailer, which rolls downhill. He is not stealing or damaging property.  
  ↳ notes: The activity is not suspicious, but rather an accident involving the owner.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0:04-0:09 - A man falls and is dragged by a rolling ATV and trailer down a hill. This is a clear accident and potential injury.  
  ↳ notes: The man is clearly in a dangerous situation, being dragged by a vehicle, which constitutes an emergency.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert when help is wanted_  
  ↳ 0:04-0:09 - A man is dragged by an ATV. He does not explicitly call for help, but the situation is dangerous.  
  ↳ notes: While the situation is dangerous, the man does not vocalize a need for help. An 'ask' is appropriate to verify if he needs assistance after the incident.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0:03-0:14 - The person involved in the incident is an adult male, not a child.  
  ↳ notes: The person is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0:03-0:14 - The person involved appears to be a middle-aged adult, not an elder.  
  ↳ notes: The person does not exhibit characteristics of an elder.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0:00-0:14 - The man is in the yard, not at the door. The camera view is of a yard/driveway area, not a door.  
  ↳ notes: The event occurs far from any visible door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s27  `716.3–731.3s`  — A mail truck enters a driveway, backs up, appears to make contact with a parked delivery truck, and then drives away.
- ⏱ **rising edge** @ `716.34s` · A mail truck is present.  
  ↳ _The mail truck enters the frame from the left._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `716.37s` · A mail truck makes contact with a delivery truck.  
  ↳ _A distinct thud sound is heard, and the mail truck visibly jolts as it backs up towards the delivery truck._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `716.49s` · The mail truck leaves the scene.  
  ↳ _The mail truck drives out of the frame to the left._  → confirm/adjust:  `t = ______`
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _Alert on suspicious activity_  
  ↳ At 0:03, a clear thud sound is heard, and the mail truck visibly jolts. The mail truck then quickly drives away without any visible inspection or interaction.  
  ↳ notes: The sound and jolt suggest an impact, and the mail truck's immediate departure without stopping is unusual. However, it's not clear if significant damage occurred or if the driver was aware of the impact, making it ambiguous.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ No people are visible in the clip. While there was a potential minor collision between two vehicles, there is no evidence of injury, fire, or significant property damage that would constitute an emergency.  
  ↳ notes: The sound of impact might initially suggest an emergency, but the visual context shows no immediate danger to life or property.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ No people are visible, and no sounds of distress or calls for help are heard.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ No children are visible in the clip.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ No elders are visible in the clip.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ The camera view is of a driveway, and no individuals approach or are present at a door.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s28  `767.1–771.1s`  — A person on a porch quickly drops to their knees, startled by a squirrel, then stands up. Another person is present.
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _someone is at the door_  
  ↳ Person 1 and Person 2 are visible on the porch from 0.0-4.0.  
  ↳ notes: Two adults are present on the porch near the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _suspicious activity_  
  ↳ Person 1 falls/kneels (0.5-1.0) but quickly recovers. Audio at 0:02.0-0:02.9 indicates a squirrel caused the reaction, not suspicious activity.  
  ↳ notes: The initial fall might appear suspicious, but the quick recovery and audio explanation clarify it was a harmless reaction to an animal.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _emergency or someone who needs help_  
  ↳ Person 1 falls/kneels (0.5-1.0) but immediately recovers and stands up (3.0-4.0). Audio at 0:02.0-0:02.9 explains the fall was due to being startled by a squirrel, not an injury or emergency.  
  ↳ notes: The fall could initially trigger an alert, but the quick recovery and audio context negate any actual emergency or need for help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _help is wanted_  
  ↳ No verbal or non-verbal cues from either person indicating a desire for help.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _child is in danger_  
  ↳ No children are visible in the clip.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _elder is in danger_  
  ↳ The two visible adults do not appear to be elderly, and neither is in danger.  
  ↳ notes:   → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s29  `822.1–828.1s`  — A person (Person A) is swinging in the yard. They lose their grip and fall to the ground. Another person (Person B) who was on the porch walks out towards Person A, calling out to them.
- ⏱ **rising edge** @ `823.61s` · Person A falls to the ground  
  ↳ _Person A loses grip on the swing and lands on the ground._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `823.61s` · Person A is on the ground  
  ↳ _Person A falls from the swing._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `824.61s` · Person B walks towards Person A  
  ↳ _Person B steps off the porch and walks towards Person A._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `827.61s` · Person B speaks  
  ↳ _Person B says 'Whoa! Really?'_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _suspicious activity_  
  ↳ 0.0-6.0: A person is swinging, falls, and another person comes to check on them. This is a playful incident, not suspicious.  
  ↳ notes: The activity is clearly recreational and harmless.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _emergency or someone who needs help_  
  ↳ 1.5: Person A falls from the swing. 5.5: Person B says 'Whoa! Really?' in a surprised but not alarmed tone. Person A does not show signs of distress or injury.  
  ↳ notes: While a fall can be serious, the context (playful activity, casual reaction from Person B, lack of distress from Person A) indicates it's not an emergency. The initial visual of a fall could be ambiguous, but the audio and subsequent actions clarify it's not an emergency.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _help is wanted_  
  ↳ 0.0-6.0: No verbal or non-verbal cues from Person A or Person B indicating help is wanted.  
  ↳ notes: No request for help is observed.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _child in danger_  
  ↳ 0.0-6.0: Person A appears to be an adult or older teenager, not a young child. Even if a child, the fall does not appear to cause severe danger, and an adult (Person B) is present to supervise/assist.  
  ↳ notes: Person A's stature suggests they are not a young child. The fall is not severe enough to indicate immediate danger, and an adult is present.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _elder in danger_  
  ↳ 0.0-6.0: Neither Person A nor Person B appear to be elders.  
  ↳ notes: No elders are present in the scene.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _someone is at the door_  
  ↳ 0.0-6.0: No one approaches the door or rings a doorbell. The activity takes place in the yard and on the porch steps, away from the door itself.  
  ↳ notes: The individuals are not interacting with the door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s30 — ⚠ DROPPED: proposed `872.5–872.6s` is out of range (video is 1146.6s).

### eval3_s31 — ⚠ DROPPED: proposed `872.6–872.6s` is out of range (video is 1146.6s).

### eval3_s32 — ⚠ DROPPED: proposed `872.6–872.7s` is out of range (video is 1146.6s).

### eval3_s33 — ⚠ DROPPED: proposed `906.9–907.2s` is out of range (video is 1146.6s).

### eval3_s34 — ⚠ DROPPED: proposed `907.2–907.2s` is out of range (video is 1146.6s).

### eval3_s35 — ⚠ DROPPED: proposed `907.2–907.2s` is out of range (video is 1146.6s).

### eval3_s36  `965.5–975.5s`  — A person on a riding lawnmower drives into a black Ford Explorer parked in a driveway.
- ⏱ **rising edge** @ `969.47s` · A riding lawnmower collides with a parked car.  
  ↳ _The lawnmower makes contact with the black Ford Explorer at 0:04 and scrapes along its side until 0:07._  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `969.47s` · Property damage occurs to a parked car.  
  ↳ _The riding lawnmower scrapes against the side of the black Ford Explorer, indicating potential damage from 0:04 to 0:07._  → confirm/adjust:  `t = ______`
- 🚦 **FIRE** (clear-fire, conf HIGH) · goal: _suspicious activity_  
  ↳ At 0:04, a person on a riding lawnmower drives into the side of a black Ford Explorer, scraping along it until 0:07.  
  ↳ notes: The collision with a parked vehicle, whether accidental or intentional, constitutes suspicious activity as it involves potential property damage.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **ASK** (ambiguous-ask, conf MED) · goal: _emergency OR person needs help_  
  ↳ At 0:04, a person on a riding lawnmower drives into the side of a black Ford Explorer. The person remains seated and continues to operate the mower, but the collision itself could indicate a medical emergency or loss of control.  
  ↳ notes: The person doesn't show clear signs of distress after the collision, but the incident itself is unusual enough to warrant checking if they are okay or if it was an uncontrolled event.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _help is wanted_  
  ↳ The person operating the lawnmower does not make any gestures or vocalizations indicating a request for help.  
  ↳ notes: No explicit request for help is observed.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _child in danger_  
  ↳ The person operating the lawnmower is an adult. No children are visible in the scene.  
  ↳ notes: The individual is clearly an adult.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _elder in danger_  
  ↳ The person operating the lawnmower appears to be an adult, but their age cannot be definitively determined as an elder. No specific danger to an elder is apparent beyond the general collision.  
  ↳ notes: The person's age is not clearly elderly, and the danger isn't specific to an elder's vulnerability.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _someone at the door_  
  ↳ The camera view shows a driveway and yard, not a door. No one approaches a door.  
  ↳ notes: The scene does not include a door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

### eval3_s37  `1024.6–1062.6s`  — A man lays a blue tarp on a driveway, then a dump truck empties a load of wood chips onto the tarp and driveway.
- ⏱ **rising edge** @ `1024.57s` · Person present  
  ↳ _Man appears in frame_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `1024.57s` · Vehicle present  
  ↳ _Dump truck is visible at the start_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `1051.57s` · Material being dumped from truck  
  ↳ _Truck bed starts to lift and material falls_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `1052.57s` · Driveway obstructed  
  ↳ _Wood chips begin to cover the driveway_  → confirm/adjust:  `t = ______`
- ⏱ **rising edge** @ `1052.57s` · gate_cases  
  ↳ _Wood chips begin to cover the driveway_  → confirm/adjust:  `t = ______`
- 🚦 **SILENT** (corrected-false, conf HIGH) · goal: _Alert on suspicious activity_  
  ↳ 0:00-0:09: Man lays tarp. 0:27-0:38: Truck dumps wood chips. This appears to be a planned delivery of landscaping material.  
  ↳ notes: Initially, a large truck and a person might seem unusual, but the deliberate action of laying a tarp and then dumping landscaping material indicates a scheduled delivery, not suspicious activity.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert on an emergency or someone who needs help_  
  ↳ 0:00-0:38: No signs of distress, injury, or emergency. The actions are deliberate and appear to be a routine delivery.  
  ↳ notes: The activity is a routine delivery of landscaping material. No one is in danger or needs help.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert when help is wanted_  
  ↳ 0:00-0:38: No verbal or non-verbal cues indicating a request for help.  
  ↳ notes: The activity is a routine delivery. No indication of help being wanted.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if a child is in danger_  
  ↳ 0:00-0:38: No children are present in the video.  
  ↳ notes: Only an adult male is seen, and he is not in danger.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Alert if an elder is in danger_  
  ↳ 0:00-0:38: The person visible is an adult, but not clearly an elder, and is not in danger.  
  ↳ notes: The adult male appears to be of working age and is performing a task safely.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask
- 🚦 **SILENT** (clear-fire, conf HIGH) · goal: _Tell me when someone is at the door_  
  ↳ 0:00-0:38: The activity is occurring in the driveway, not near a door. The camera angle is from above, looking down a driveway.  
  ↳ notes: The activity is confined to the driveway, far from any apparent door.  → keep ☐ / flip to ☐ fire ☐ silent ☐ ask

