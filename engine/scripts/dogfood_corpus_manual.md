# Dogfood corpus — manual walkthrough (Qwen / OpenCode)

Paste this in by hand, turn by turn, into a live `qwen-code` session, so you
can read Qwen's real replies and watch `/memory` fill in as you go.

## How to run this

1. **Start a brand-new OpenCode session before EACH numbered conversation
   below, and exit OpenCode (so the session actually closes) once you finish
   a conversation's turns.** This isn't just tidiness -- it matters for two
   real reasons:
   - The write hook only fires once per session, at exit (`dispose`) -- a
     conversation isn't consolidated into the graph until you actually quit.
   - Several probes at the end are specifically testing whether the **graph**
     resurfaces a fact appropriately. If everything ran in one long session,
     Qwen's own live chat context (not the graph) would explain any recall,
     and the test would stop measuring what it's meant to measure.
2. **Only paste the bold `Turn` blocks** (in code fences, so they're easy to
   select and copy cleanly). The italicized blockquote under each turn is
   just a *reference* of what the scripted version of this conversation
   assumed the assistant might say -- it's there so you know the intended
   direction, not something to paste. Qwen's actual reply will differ in
   wording; that's fine, just read it and move to the next turn regardless of
   exactly how it phrased things.
3. Run the conversations **in the numbered order** -- this is one persona's
   memory graph accumulating over weeks, and several later conversations
   (job change, corrections, callbacks) assume earlier ones already landed.
4. After all conversations are done and each session has been closed (so
   every write has actually fired), move to the **Probes** section at the
   bottom. Same rule applies: **each probe should be asked in its own fresh
   session** -- the point is to see what the graph alone surfaces with zero
   live chat history to lean on.
5. Check `/memory` (or `http://127.0.0.1:8765/viewer`) whenever you want to
   see the graph directly instead of only through what gets injected.

---

## 1. `intro_identity`

**Turn 1:**
```
Hey, first time talking to you properly -- I'm Devon Ruiz, I go by Devon. I'm a backend software engineer, currently based in Austin.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Good to meet you, Devon! Backend engineering in Austin -- what's your day-to-day look like, and what got you into backend work specifically?

**Turn 2:**
```
I like systems that fail predictably, I guess. I've been doing this for about seven years now.
```

---

## 2. `family_and_partner`

**Turn 1:**
```
Random context for later: my partner's name is Jules, we've been together four years. My younger sister Nat lives in Portland and we talk most weekends.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Got it -- Jules as your partner, and Nat as your sister in Portland. Nice that you two keep up a regular weekend call, that's not always easy to maintain.

**Turn 2:**
```
Yeah it's basically the one non-negotiable thing in my calendar.
```

---

## 3. `job_at_vantage`

**Turn 1:**
```
I work at a company called Vantage Robotics, on the perception team. Been there about three years.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Vantage Robotics, perception team, three years in -- solid tenure. What kind of perception problems are you working on there, computer vision, sensor fusion, something else?

**Turn 2:**
```
Mostly sensor fusion for warehouse robots. It's less glamorous than it sounds but the problems are genuinely hard.
```

---

## 4. `manager_priya`

**Turn 1:**
```
My manager at Vantage is Priya Subramaniam. She's been really good about protecting the team from scope creep.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That's a valuable trait in a manager, especially on a hard technical team where scope creep can quietly eat months. How long has Priya been your manager?

**Turn 2:**
```
Since I joined, so three years. She's the reason I've stayed this long honestly.
```

---

## 5. `debugging_session_noisy`

**Turn 1:**
```
Been chasing a nasty bug in the sensor fusion pipeline at work today. Getting this: `KalmanFilter.update() -> ValueError: covariance matrix not positive semi-definite at t=142.7883s`. Traceback points into `perception/fusion/kalman.py:212`, inside `_update_covariance`. Any idea what typically causes this?
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That error usually means numerical drift is eating away at your covariance matrix's symmetry over many update steps -- floating point rounding can slowly break positive semi-definiteness even when the math is correct in theory. The standard fixes are enforcing symmetry after each update (P = (P + P.T) / 2) or switching to a square-root/UKF formulation that's more numerically stable. Do you know roughly how many update cycles run before it fails?

**Turn 2:**
```
Good call -- turned out to be clock drift between two sensors feeding slightly desynced timestamps into the update step, which compounded over a few thousand cycles. Adding the symmetry fix plus a timestamp sanity check cleared it up. Took most of the afternoon to track down, that clock-drift-from-desynced-sensors pattern is one I want to remember for next time.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Glad you tracked it down -- desynced sensor timestamps causing slow covariance drift is a subtle one, definitely worth keeping in your back pocket for the next time something looks like a math bug but is actually a timing bug.

---

## 6. `project_lighthouse_stack`

**Turn 1:**
```
Side project question -- I'm building a personal blog/portfolio site, calling it Lighthouse. Torn between Astro and Next.js for it. It's mostly static content, a handful of blog posts, no real interactivity.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> For mostly-static content with minimal interactivity, Astro is the stronger fit -- it ships zero JS by default and only hydrates the islands you explicitly mark, whereas Next.js carries more runtime weight you won't use for a static blog. If you ever need heavier client-side state later, Next.js scales up more easily, but that doesn't sound like Lighthouse's shape.

**Turn 2:**
```
Yeah that matches what I was leaning toward. Going with Astro for Lighthouse then -- the content collections feature looks like a good fit for the blog posts too.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Good call. Astro's content collections give you type-safe frontmatter out of the box, which will save you some pain once you have more than a couple posts.

---

## 7. `pet_biscuit`

**Turn 1:**
```
Unrelated -- my dog Biscuit just turned six. He's a very anxious corgi who hates the mail carrier specifically.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Happy birthday to Biscuit! Six years of mail-carrier vigilance is a real commitment. Corgis do tend to be alert little guys.

---

## 8. `hobby_climbing_intro`

**Turn 1:**
```
I've gotten really into bouldering the last year. I go with my friend Theo three times a week at a gym near my place.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Three times a week is a serious habit, not just a hobby at that point. What's kept you and Theo coming back -- the physical challenge, the problem-solving side, the social part?

**Turn 2:**
```
Honestly the problem-solving. Every route is basically a puzzle. Theo's way better than me though, he's been climbing for a decade.
```

---

## 9. `ephemeral_car_trouble`

**Turn 1:**
```
Ugh, my car wouldn't start this morning, had to get a jump from a neighbor and I'm going to be late to everything today.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That's a rough start to the day. Was it the battery, or something else going on? If it's an older battery, might be worth getting it tested before it strands you somewhere less convenient.

**Turn 2:**
```
Probably the battery, it's been slow to turn over for a couple weeks. Going to deal with it this weekend, I hope.
```

---

## 10. `allergy_claim`

**Turn 1:**
```
Quick note for anything food-related -- I'm allergic to shellfish, so keep that in mind if you ever suggest restaurants or recipes.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Noted, no shellfish. Thanks for flagging that up front -- I'll steer clear of it in any restaurant or recipe suggestions.

---

## 11. `book_recommendation`

**Turn 1:**
```
Just finished Project Hail Mary and honestly it might be my favorite book of the last few years.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Project Hail Mary is a great pick -- Andy Weir really nails the problem-solving-under-pressure tone again. Was it the science puzzles or the Rocky relationship that got you most?

**Turn 2:**
```
The Rocky relationship, no contest. Didn't expect to get that attached to it.
```

---

## 12. `general_knowledge_oauth`

**Turn 1:**
```
Can you explain how OAuth2's authorization code flow actually works end to end?
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Sure -- the client redirects the user to the authorization server, the user logs in and consents, the server redirects back with a short-lived authorization code, and then the client exchanges that code (plus its client secret) for an access token via a back-channel request. The key security property is that the access token never passes through the browser, only the code does.

**Turn 2:**
```
Got it, that makes sense, thanks.
```

---

## 13. `ambiguous_alex_friend`

**Turn 1:**
```
My friend Alex from college is visiting next month. We haven't seen each other in person in like two years.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That's exciting, two years is a long time between visits. Do you two have anything specific planned, or mostly just catching up?

**Turn 2:**
```
Mostly catching up. Alex and I were roommates sophomore and junior year, so there's a lot of history there.
```

---

## 14. `time_bound_marathon`

**Turn 1:**
```
I'm training for a half marathon in Austin on October 18th. Currently about eight weeks out.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Eight weeks out is a good spot to be building your long runs up steadily. How's the training going so far -- on pace, or feeling behind?

**Turn 2:**
```
On pace I think. Longest run so far is nine miles, feeling okay.
```

---

## 15. `preference_coffee_diet`

**Turn 1:**
```
Random one -- I'm vegetarian, and my go-to coffee order is an oat milk cortado.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Vegetarian, oat milk cortado -- noted. A cortado's a good choice if you want the espresso to actually come through instead of getting buried in milk.

---

## 16. `podcast_mention`

**Turn 1:**
```
I've been listening to a podcast called Search Engine on my runs lately, really well produced.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Search Engine's a good one -- PJ Vogt's got a knack for picking questions that sound small and turning them into something surprisingly deep. What's been the best episode for you so far?

**Turn 2:**
```
The one about why gas station coffee is bad, weirdly. Very not-small by the end.
```

---

## 17. `creative_writing_poem`

**Turn 1:**
```
Can you write me a short four-line poem about rain for a card I'm making?
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Sure: "The rain does not ask permission to fall, / it simply arrives and softens it all -- / the streets, the roofs, the tired afternoon light, / washed clean and quiet by morning's first sight."

**Turn 2:**
```
That's really nice, thank you.
```

---

## 18. `allergy_correction`

**Turn 1:**
```
Hey, actually I need to correct something I told you before -- I said I'm allergic to shellfish, but that's not quite right. I just really don't like shrimp, no actual allergy. My bad for the confusion.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Good to know, thanks for the correction -- no allergy, just a shrimp dislike. I'll keep that distinction in mind, it does change what's actually off the table versus what's just a preference.

---

## 19. `hackathon_distractor`

**Turn 1:**
```
I'm thinking about signing up for a weekend web dev hackathon here in Austin. Never done one before.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That could be a lot of fun, especially as a first one -- low stakes, high learning. Are you going in with a team already, or planning to find one there? Also worth thinking about ahead of time: do you want to optimize for finishing something polished, or for trying something you wouldn't normally get to touch at work?

**Turn 2:**
```
No team yet, might just show up solo and see who I match with. Leaning toward trying something outside my usual backend comfort zone.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Solo entry is a great way to meet people you'd never end up paired with otherwise. If you want to stretch outside backend, a lot of hackathon teams are short on whoever can wire up a working frontend fast, so that's often an easy way to pair up quickly.

---

## 20. `old_car_story`

**Turn 1:**
```
Saw a beat-up Honda Civic on the road today that looked exactly like my first car in college, a 2011 Civic I named Gerald for no good reason.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Gerald the Civic is a great car name. Do you remember why 2011 Civics specifically stuck with you, or was Gerald just what you could afford at the time?

**Turn 2:**
```
Definitely just what I could afford. But that car survived four years of very questionable driving decisions.
```

---

## 21. `childhood_sailing`

**Turn 1:**
```
Random memory that came up today -- I grew up sailing with my dad on weekends, small lake near where I grew up. Haven't done it in years though.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That sounds like a nice thing to have had growing up. What made you stop -- just life getting busier, or moving away from the lake?

**Turn 2:**
```
Mostly moving for college and then work. I keep meaning to get back into it someday.
```

---

## 22. `samuel_intro`

**Turn 1:**
```
My cousin Samuel is coming through Austin next week on a work trip, we're going to grab dinner.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Nice, a work-trip dinner with family is always a good excuse to catch up. Is Samuel someone you see often, or is this more of a rare crossing-paths thing?

**Turn 2:**
```
Pretty rare honestly, maybe once a year. He travels a ton for his job.
```

---

## 23. `move_new_city`

**Turn 1:**
```
Big update -- Jules and I are moving to Denver in a couple months. Mostly for Jules's job, but I'm excited about it too.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That's a big change! Denver's a solid pick if you like being close to mountains. How are you feeling about leaving Austin after all this time here?

**Turn 2:**
```
Mixed feelings. Austin's been home for a while, but change feels good right now. Going to miss the climbing gym here the most, honestly.
```

---

## 24. `ambiguous_alex_coworker`

**Turn 1:**
```
Started meeting some new people ahead of the Denver move -- there's an engineer at Nimbus named Alex Park who's going to be on my new team. Different Alex from my college friend, just a funny coincidence.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Good coincidence to flag, that would get confusing fast otherwise. What's your first impression of Alex Park so far?

**Turn 2:**
```
Seems sharp, asked good questions in the intro call. We'll see once I'm actually working with them day to day.
```

---

## 25. `job_change_nimbus`

**Turn 1:**
```
So the Denver move is actually because I took a new job -- I'm leaving Vantage Robotics and joining Nimbus Health as a senior backend engineer. Starts the same week we move.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Congratulations, that's a meaningful jump -- new company, new domain, new city all at once. What pulled you toward Nimbus Health specifically?

**Turn 2:**
```
The problem space, mostly. Health data infrastructure feels higher-stakes than warehouse robots in a good way. Bittersweet leaving Vantage after three years though.
```

---

## 26. `manager_callum`

**Turn 1:**
```
Met my new manager at Nimbus, Callum Reyes, in an onboarding call. Seems very direct, in a good way so far.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Direct managers can be a nice change of pace if you like knowing exactly where you stand. How does Callum compare to Priya so far, or is it too early to tell?

**Turn 2:**
```
Too early to tell really, but the directness reminds me a bit of Priya actually. Hoping that pattern holds.
```

---

## 27. `acknowledged_impostor_syndrome`

**Turn 1:**
```
I keep feeling like I got the Nimbus offer by luck, not because I'm actually qualified for a senior role, even though I know objectively that's not true.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That gap between what you know objectively and what you feel has a name -- it's a pretty textbook case of impostor syndrome, and it's especially common right after a level-up like this one. It doesn't mean anything's actually wrong with your qualifications.

**Turn 2:**
```
Yeah, exactly, that's impostor syndrome. I've heard the term before but never really applied it to myself until just now.
```

---

## 28. `emotional_argument_roommate`

**Turn 1:**
```
Had a dumb argument with my old roommate Theo this week about who was supposed to handle the gym membership cancellation before I move. We sorted it out but it was annoying in the moment.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Moving logistics have a way of surfacing exactly that kind of friction. Glad it got sorted out -- was it a quick resolution once you actually talked it through?

**Turn 2:**
```
Yeah, five minute conversation once we both stopped being annoyed. Classic miscommunication, nothing serious.
```

---

## 29. `duolingo_streak`

**Turn 1:**
```
Hit a 400 day streak on Duolingo learning Spanish today, small win but I'm pretty proud of it.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> 400 days is genuinely impressive, that's real consistency. How's your actual conversational Spanish coming along alongside the streak?

**Turn 2:**
```
Slowly but surely. Reading's way ahead of speaking for me.
```

---

## 30. `fiddle_leaf_fig`

**Turn 1:**
```
My fiddle leaf fig finally pushed out a new leaf after basically sulking for two months. Small victories.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Fiddle leaf figs are famously dramatic about that, so a new leaf is a real win. What ended up turning it around, more light, different watering, or just patience?

**Turn 2:**
```
Honestly just patience and leaving it alone. I think I was fussing over it too much before.
```

---

## 31. `timbers_sports`

**Turn 1:**
```
Portland Timbers won again this weekend, my sister Nat's been sending me way too many celebratory texts.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Nat sounds like a committed fan. Are you into the Timbers too, or is this more secondhand enthusiasm via your sister?

**Turn 2:**
```
Mostly secondhand, but it's rubbing off on me a little if I'm honest.
```

---

## 32. `tamale_recipe`

**Turn 1:**
```
Made my grandmother's tamale recipe this weekend for the first time solo, no supervision. They actually turned out well.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That's a real milestone, family recipes carry a lot of pressure to get right the first time alone. What was the trickiest part without supervision?

**Turn 2:**
```
Getting the masa consistency right, definitely. Took two batches to get it where I remembered it should feel.
```

---

## 33. `samuel_callback`

**Turn 1:**
```
Sam texted me a photo from his latest work trip, this time in Singapore. That guy travels more than anyone I know.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Singapore's a good one to add to the list. Is Sam's travel mostly client visits, or does his role just have him bouncing around constantly regardless of specific trips?

**Turn 2:**
```
Pretty much constant regardless. I don't think he's been in one place for more than three weeks straight all year.
```

---

## 34. `hobby_climbing_callback`

**Turn 1:**
```
Going to miss climbing with Theo most out of everything once we're actually in Denver. We've found a new gym to check out there at least.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Denver actually has a strong climbing scene, so that transition might go smoother than you'd expect even without Theo there. Have you looked at whether the new gym has a similar bouldering-heavy setup to what you're used to?

**Turn 2:**
```
Yeah, looks pretty similar from photos. Still won't be the same without Theo roasting my technique though.
```

---

## 35. `unrelated_smalltalk`

**Turn 1:**
```
Hey, how's it going today?
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> Doing well, thanks for asking! How about you -- anything on your mind today, or just checking in?

**Turn 2:**
```
Just checking in, nothing much going on. Kind of a slow day.
```

---

## 36. `dentist_appointment`

**Turn 1:**
```
Have a dentist appointment tomorrow morning that I'm dreading for no real reason, just general dentist-visit anxiety.
```

> *(reference reply, don't paste — Qwen's real answer will differ)*
> That's a really common feeling even when there's nothing actually wrong. Is it a routine cleaning, or is there something specific prompting this one?

**Turn 2:**
```
Just routine cleaning. I'll probably feel silly for dreading it once it's over, like always.
```

---


---

## Probes — run each in a brand-new session

For each one: start a fresh OpenCode session, paste only the question, and
compare what actually gets injected/said against the `expect` note.

### Probe 1

```
I'm thinking about signing up for that web dev hackathon next month, any tips on how to prep?
```

*expect*: REGRESSION CHECK: this echoes hackathon_distractor almost verbatim. The Astro/Lighthouse fact must NOT appear here -- if it does, that's the Karpathy-tweet failure mode (a tangential fact bleeding into an unrelated turn) and the graph is behaving like a flat summary instead of context-triggered recall.

### Probe 2

```
What stack did I end up going with for Lighthouse again?
```

*expect*: Should surface Astro + the content-collections detail via the snippet, since Lighthouse is named explicitly here.

### Probe 3

```
Where do I work these days, and who's my manager?
```

*expect*: Should surface Nimbus Health / Callum Reyes as CURRENT -- Vantage/Priya should be superseded, not returned as if still active (mutable WORKS_AT + one_to_one HAS_MANAGER supersede check).

### Probe 4

```
Remind me about Alex.
```

*expect*: Ambiguity check: there are two distinct Alexes (college friend, Nimbus coworker Alex Park). Correct behavior is either surfacing both distinctly or the retriever asking-shaped context reflecting two separate nodes -- NOT one merged Alex with contradictory details.

### Probe 5

```
Am I allergic to anything I should watch out for with food?
```

*expect*: Should reflect the CORRECTED state (shrimp dislike, no real allergy) -- not the original shellfish-allergy claim. Tests contradiction/supersede on a PREFERENCE-flavored fact.

### Probe 6

```
How's my half marathon training going?
```

*expect*: Time-bound fact recall -- should mention the October 18th Austin half marathon and roughly the training status mentioned (nine mile long run).

### Probe 7

```
What's the deal with my car?
```

*expect*: Ephemeral fact -- should surface the battery/jump-start incident. Note: real decay won't be visible yet in a same-day run; this is a retrieval-presence check, not a decay check.

### Probe 8

```
Can you explain OAuth2 again?
```

*expect*: Should NOT surface any injected [MEMORY] block distinct from actual OAuth2 knowledge -- the earlier OAuth2 conversation was a pure Q&A with no durable content, so nothing graph-worthy should have been extracted from it in the first place.

### Probe 9

```
What's my dietary preference for restaurant recommendations?
```

*expect*: Should surface vegetarian (a real durable preference), distinct from the shellfish/shrimp correction above -- two separate food-related facts, don't let one clobber the other.

### Probe 10

```
Tell me about Sam.
```

*expect*: Abbreviation-merge check: 'Samuel' (cousin, intro'd by full name) and later 'Sam' (same person, casual) should resolve to ONE node, not two.

### Probe 11

```
How's the climbing going, and who do I usually climb with?
```

*expect*: Should surface Theo + bouldering + the gym context, ideally reflecting both the original intro and the later Denver-move callback (co-episode / reinforcement check).

### Probe 12

```
What was that book I couldn't stop talking about recently?
```

*expect*: Should surface Project Hail Mary, ideally with the Rocky-relationship detail from the snippet rather than just the title.

### Probe 13

```
Do I have any pets?
```

*expect*: Should surface Biscuit the corgi.

### Probe 14

```
What's going on with my living situation and family right now?
```

*expect*: Broad recall -- should pull together Jules (partner), the Denver move, and possibly Nat (sister) without needing any of those named explicitly in the query. This is the multi-hop / spreading-activation stress test of the whole probe set.

### Probe 15

```
Have I mentioned feeling unsure about myself at work lately?
```

*expect*: Should surface the impostor-syndrome conversation -- tests the 'acknowledged agent concept' extraction path (assistant named the concept, user affirmed it) rather than a verbatim keyword match.

### Probe 16

```
What was that tricky bug I spent an afternoon chasing down at Vantage?
```

*expect*: Coding-session durability check: the surrounding stack trace / file path / error message noise should NOT have spawned garbage nodes, but the actual debugging lesson (desynced sensor clocks causing covariance drift) is exactly the kind of hard-won, one-off-but-worth-remembering fact the durability-threshold discussion is about. Under the CURRENT threshold it may well have been dropped as 'not durable' -- if so, that's a concrete before/after data point for the phase-2 threshold change, not a bug in this run.
