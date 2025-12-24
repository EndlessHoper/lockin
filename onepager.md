# Lock In - AI Study Buddy

**One-liner:** A study buddy who's on the call with you — watching via webcam and screen, silently vibing, speaking up when you slip.

---

## The Core Loop

You call your study buddy. They pick up. Now you're on a FaceTime-style call together — except they're an AI with a personality you chose. They see your webcam and your screen. They're just... there with you. Quietly present.

When you slip — phone in hand, Twitter open, glazed-over eyes — they just speak up. No notification. No popup. They're already on the call. Like a real friend would.

> "Bro. I can see Instagram. The derivatives aren't gonna differentiate themselves."

They know what you're working on because you told them when you called. They track your focus. They celebrate wins. They give you a debrief when you hang up. It's the study buddy you wish you had.

---

## Key Experiences

### 1. Start a Session ("Call your buddy")
- You initiate the "call" — hit the button, your buddy picks up
- They greet you, ask what you're working on (voice or text)
- Set duration (or open-ended "until I hang up")
- Pick your vibe: deep focus, chill study, cram mode
- Once you start, you're "on the call" together for the whole session

### 2. Passive Monitoring
- Webcam: detects phone pickup, looking away, head down (asleep?), zoned out stare
- Screen capture: detects off-task apps/sites (social media, YouTube, games, etc.)
- Sampling every few seconds, runs local, not a battery hog

### 3. The Intervention
- They're already on the call — they just speak up
- No notification, no popup, no "incoming call" — they're already there with you
- Feels like a real person on FaceTime noticing you got distracted
- Personalized to your session context:
  - "Chapter 4 isn't going to read itself, let's go"
  - "You said this essay is due tomorrow. Why are we on Reddit?"
- Escalation: first gentle, then roast mode if you keep slipping
- Option to say "I'm on a break" (voice command or button) to pause monitoring
- They can also speak up positively: "20 minutes locked in, nice" or just ambient presence

### 4. Session End / Hang Up
- When you end the call, buddy gives you a debrief (TTS)
- "Solid session. 47 minutes focused, only checked your phone twice. Better than yesterday. Same time tomorrow?"
- Summary: focus time, distraction count, longest streak
- Streaks, history, trends over time

---

## Personality System

Choose your accountability partner:

| Persona | Vibe | Sample Line |
|--------|------|-------------|
| Drill Sergeant | Intense, no excuses | "Phone DOWN. You think success takes breaks? MOVE." |
| Disappointed Parent | Guilt-trip master | "I'm not mad. I'm just... I expected more from you." |
| Hype Beast | Your biggest fan | "Yo you got this, don't let the timeline steal your dreams king" |
| British Friend | Passive-aggressive | "Oh, checking Twitter again, are we? Lovely. Brilliant. Cool." |
| Chill Tutor | Gentle nudges | "Hey, noticed you drifted. Wanna take a real break or get back to it?" |
| Custom | User-defined | Paste any persona description |

Personality affects:
- TTS voice selection (pitch, speed, accent if available)
- Language and tone of interventions
- How quickly they escalate

---

## Context Awareness

Before or during session, user provides:
- What they're working on (free text or voice): "Calculus problem set 3"
- Deadline pressure: "Due tomorrow" vs "just reviewing"
- Allowed apps/sites (whitelist): "YouTube is okay, I'm watching lectures"

This lets the AI:
- Reference the actual task in callouts
- Adjust strictness based on deadline
- Not yell at you for being on an allowed site

---

## Pomodoro / Timer Integration

Optional structure:
- Classic 25/5 pomodoro
- Custom intervals
- Or just free-form "monitor me until I stop"

During breaks:
- Buddy knows you're on break, won't intervene
- Optional: "Break's over in 1 minute" nudge
- Screen/webcam still on but passive

---

## Platform: macOS Native App

**Why native over Chrome extension:**
- Webcam + screen capture permissions in one place
- Works across all apps, not just browser
- Can detect phone pickup even when not in browser
- More reliable background process
- Cleaner "call" UI experience

**Tech stack options:**
- Tauri (Rust + web frontend) — lightweight, modern
- Electron — heavier but faster to build if you know it
- Swift/SwiftUI — most native feel, steeper curve

**Two-model architecture:**

*1. Local vision model (detection):*
- Runs locally via llama.cpp, MLX, or similar
- SmolVLM, MobileVLM, LLaVA variant, etc.
- Samples webcam + screen every 2-5 seconds
- Simple classification: "Is this person distracted? What are they doing?"
- Lightweight, fast, private — just answers "distracted: yes/no, reason: phone/twitter/zoned out"

*2. API model (callout generation):*
- Only called when distraction detected
- Model-agnostic: Opus 4.5, GPT-5.2 medium, Grok, etc. (user can pick or we default)
- Receives rich context:
  - Persona system prompt (Drill Sergeant, Disappointed Parent, etc.)
  - Session context (what they're working on, deadline, how long they've been focused)
  - Distraction type (phone, specific app/site, zoned out, etc.)
  - History (how many times they've slipped this session, what was said before)
- Generates the actual callout line with personality and context
- Keeps responses short and punchy (one or two sentences max)

*Why split it:*
- Local model runs constantly without API costs or latency
- API model only fires on distraction events (maybe 0-10x per session)
- API model is way better at humor, personality, context-aware roasts
- Users can bring their own API key / pick their preferred model

*Example API payload when distraction detected:*
```json
{
  "persona": "drill_sergeant",
  "persona_prompt": "You are a military drill sergeant. Intense, no excuses, short barking commands. You believe in the user but won't let them slack.",
  "session": {
    "task": "Econ essay on market failures",
    "deadline": "tomorrow",
    "duration_so_far": "32 minutes",
    "focus_streak": "12 minutes",
    "distractions_this_session": 2
  },
  "distraction": {
    "type": "app",
    "detail": "Twitter/X open in browser",
    "timestamp": "..."
  },
  "history": [
    { "time": "18:04", "type": "phone", "response": "Phone DOWN. Essay. NOW." },
    { "time": "18:21", "type": "zoned_out", "response": "Eyes on the screen, soldier. Market failures won't write themselves." }
  ]
}
```
→ API returns: "Twitter? TWITTER? You said this essay is due TOMORROW. Close it. Move."

**TTS:**
- System TTS for v1 (free, instant)
- Eleven Labs / OpenAI TTS for character voices later
- Could even let users clone their own voice or a friend's

---

## UX Sketches

### Main States

```
┌─────────────────────────────────────────┐
│  IDLE STATE                             │
│                                         │
│     ┌───────────────────┐               │
│     │                   │               │
│     │   [Avatar/Face]   │               │
│     │                   │               │
│     └───────────────────┘               │
│                                         │
│     "Ready when you are"                │
│                                         │
│     [ Start Study Session ]             │
│                                         │
│     Today: 2h 34m focused               │
│     Streak: 5 days 🔥                   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  SESSION SETUP                          │
│                                         │
│  What are we working on?                │
│  ┌─────────────────────────────────┐    │
│  │ "Econ essay on market failures" │    │
│  └─────────────────────────────────┘    │
│        [ 🎤 Voice input ]               │
│                                         │
│  How long?                              │
│  [ 25 min ] [ 50 min ] [ ∞ Open ]       │
│                                         │
│  Your buddy today:                      │
│  [ 🎖 Drill Sgt ] [ 😤 Parent ] [ 🔥 Hype ] │
│                                         │
│           [ Let's Lock In ]             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  ACTIVE SESSION (You're "on the call") │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │     [Buddy avatar/animation]    │    │
│  │        (looking at you)         │    │
│  │                                 │    │
│  │  ┌─────────┐                    │    │
│  │  │ Your    │                    │    │
│  │  │ webcam  │                    │    │
│  │  └─────────┘                    │    │
│  └─────────────────────────────────┘    │
│                                         │
│    Econ Essay | 32:15 | 🔥 Locked in    │
│                                         │
│  [ 🎤 Talk ] [ Break ] [ End Session ]  │
└─────────────────────────────────────────┘

When they speak up, it's just them talking on the call:

┌─────────────────────────────────────────┐
│  ACTIVE SESSION                         │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │     [Buddy avatar/animation]    │    │
│  │      (mouth moving, speaking)   │    │
│  │                                 │    │
│  │  ┌─────────┐                    │    │
│  │  │ Your    │                    │    │
│  │  │ webcam  │                    │    │
│  │  └─────────┘                    │    │
│  └─────────────────────────────────┘    │
│                                         │
│  💬 "I see Twitter open. The essay,    │
│      remember? You got this."           │
│                                         │
│    Econ Essay | 32:15 | ⚠️ Distracted   │
│                                         │
│  [ 🎤 Talk ] [ Break ] [ End Session ]  │
└─────────────────────────────────────────┘
```

---

## MVP Scope (Ship in a weekend)

**In:**
- Webcam monitoring for phone/looking away
- Screen monitoring for blacklist apps (Twitter, IG, Reddit, YouTube)
- One or two preset personalities
- Basic TTS callouts
- Simple session timer
- Session summary at end

**Out (v2+):**
- Voice input for session setup
- Custom personas
- Detailed analytics dashboard
- Break timer / pomodoro modes
- Streaks and gamification
- Screen recording review ("here's when you got distracted")
- Friends/accountability groups
- iOS/mobile companion

---

## Name Ideas

- **Lock In** — direct, meme-friendly
- **Buddy** — simple, warm
- **On Call** — plays on the FaceTime metaphor
- **Focus Call** — descriptive
- **Study Buddy AI** — SEO-friendly but generic
- **Locked** — edgy
- **Coach** — straightforward

---

## Open Questions

1. **Privacy framing:** Webcam + screen capture is sensitive. How do we communicate "all local, nothing leaves your machine" clearly?

2. **Intervention frequency:** How often is too often? Need a cooldown after each callout (30 sec? 1 min?) so it's not annoying.

3. **False positives:** What if you're just thinking and staring into space? Need a "I'm thinking" gesture or button?

4. **Break UX:** Voice-activated "I'm taking a break" or explicit button press? Voice is cooler but might misfire.

5. **Monetization (later):** Premium voices? Longer session history? Team/accountability features?

---

## Demo Video Script (for launch tweet)

1. Open app, quick setup: "Working on calc homework, 25 minutes, give me Drill Sergeant"
2. Hit "Let's Lock In" — call starts, buddy appears on screen, says "Alright, let's get it. Calc homework. I'm watching."
3. Show studying for a few seconds, buddy is just quietly there
4. Reach for phone — buddy speaks up immediately: "Phone down. The problem set. Now."
5. Put phone down, back to work
6. Open Twitter — buddy notices: "I see Twitter. You said calc. Come on."
7. End session, buddy gives debrief: "28 minutes focused, slipped twice. Better than yesterday. Same time tomorrow?"
8. "Lock In. Your AI study buddy who's actually on the call with you."

---

## Next Steps

1. [ ] Scaffold Tauri app with webcam + screen capture permissions
2. [ ] Get SmolVLM or similar running locally via llama.cpp/MLX
3. [ ] Build the detection prompt (distraction classifier)
4. [ ] Wire up system TTS
5. [ ] Basic UI: session start → active → intervention → summary
6. [ ] Record demo, ship it
