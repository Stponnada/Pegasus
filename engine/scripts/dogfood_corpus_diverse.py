"""Diverse hand-authored conversation corpus for local (Qwen-backed) dogfooding.

Unlike scripts/dogfood.py (single Roman-history persona, assistant turns
generated live by Gemini), everything here is STATIC: both user and assistant
turns are hand-written. That's deliberate for this run:

  - You (the human) are running this yourself specifically to inspect the
    exact data going into extraction and judge Qwen's output quality. Static
    transcripts mean what goes in is exactly what you read here -- no
    generation-quality confound.
  - The corpus is designed to probe the core architectural claim behind
    ditching flat-text summaries: the graph can absorb obscure, low-frequency
    facts (a tech-stack choice for one side project) that should stay SILENT
    in unrelated conversations and only surface when something actually
    triggers that node again. See `project_lighthouse_stack` /
    `hackathon_distractor` / probe "hackathon_tips" below -- that trio is the
    direct regression test for the Karpathy tweet failure mode (a stray fact
    bleeding into unrelated turns because it sits in a flat, always-injected
    summary instead of a graph that only lights up when addressed).

Categories covered (spread across ~35 conversations, one shared persona,
"Devon Ruiz", so the graph accumulates and interconnects like a real user's
would):
  - durable identity / family facts (immutable, stable)
  - a project-specific tech decision + its distractor + its recall probe
  - a job change (mutable WORKS_AT supersede) and a manager change
    (one_to_one HAS_MANAGER relational singularity)
  - ephemeral one-offs (car trouble, a dentist visit, a roommate spat)
  - a time-bound plan (marathon training with a date)
  - pure Q&A with no durable content (durability-threshold / empty-extraction
    check)
  - creative-writing help (should not leak personal facts into the graph)
  - two same-first-name people who must NOT get merged (ambiguous "Alex")
  - a stated fact later explicitly corrected (contradiction/supersede)
  - an "acknowledged agent concept" (assistant names a pattern, user affirms
    with no hedge -> lower-confidence extraction per spec)
  - an abbreviation pair (Samuel -> later "Sam")
  - a batch of small, unrelated one-off vignettes purely to grow node/edge
    count for the graph-at-scale question (pet, book, podcast, old car,
    childhood memory, recipe, streak app, houseplant, sports team)
  - pure smalltalk with zero durable content (false-positive check)
  - late "callback" conversations that reference earlier facts casually, to
    check retrieval + Hebbian reinforcement on repeat activation

Run these through scripts/run_local_dogfood.py against a live ontomem
service (Qwen via the HPC bridge, or any other backend) IN THIS ORDER --
several probes assume earlier conversations already landed in the graph.
"""

CONVERSATIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("intro_identity", [
        ("user", "Hey, first time talking to you properly -- I'm Devon Ruiz, I go by Devon. I'm a backend software engineer, currently based in Austin."),
        ("assistant", "Good to meet you, Devon! Backend engineering in Austin -- what's your day-to-day look like, and what got you into backend work specifically?"),
        ("user", "I like systems that fail predictably, I guess. I've been doing this for about seven years now."),
    ]),
    ("family_and_partner", [
        ("user", "Random context for later: my partner's name is Jules, we've been together four years. My younger sister Nat lives in Portland and we talk most weekends."),
        ("assistant", "Got it -- Jules as your partner, and Nat as your sister in Portland. Nice that you two keep up a regular weekend call, that's not always easy to maintain."),
        ("user", "Yeah it's basically the one non-negotiable thing in my calendar."),
    ]),
    ("job_at_vantage", [
        ("user", "I work at a company called Vantage Robotics, on the perception team. Been there about three years."),
        ("assistant", "Vantage Robotics, perception team, three years in -- solid tenure. What kind of perception problems are you working on there, computer vision, sensor fusion, something else?"),
        ("user", "Mostly sensor fusion for warehouse robots. It's less glamorous than it sounds but the problems are genuinely hard."),
    ]),
    ("manager_priya", [
        ("user", "My manager at Vantage is Priya Subramaniam. She's been really good about protecting the team from scope creep."),
        ("assistant", "That's a valuable trait in a manager, especially on a hard technical team where scope creep can quietly eat months. How long has Priya been your manager?"),
        ("user", "Since I joined, so three years. She's the reason I've stayed this long honestly."),
    ]),
    ("debugging_session_noisy", [
        ("user", "Been chasing a nasty bug in the sensor fusion pipeline at work today. Getting this: `KalmanFilter.update() -> ValueError: covariance matrix not positive semi-definite at t=142.7883s`. Traceback points into `perception/fusion/kalman.py:212`, inside `_update_covariance`. Any idea what typically causes this?"),
        ("assistant", "That error usually means numerical drift is eating away at your covariance matrix's symmetry over many update steps -- floating point rounding can slowly break positive semi-definiteness even when the math is correct in theory. The standard fixes are enforcing symmetry after each update (P = (P + P.T) / 2) or switching to a square-root/UKF formulation that's more numerically stable. Do you know roughly how many update cycles run before it fails?"),
        ("user", "Good call -- turned out to be clock drift between two sensors feeding slightly desynced timestamps into the update step, which compounded over a few thousand cycles. Adding the symmetry fix plus a timestamp sanity check cleared it up. Took most of the afternoon to track down, that clock-drift-from-desynced-sensors pattern is one I want to remember for next time."),
        ("assistant", "Glad you tracked it down -- desynced sensor timestamps causing slow covariance drift is a subtle one, definitely worth keeping in your back pocket for the next time something looks like a math bug but is actually a timing bug."),
    ]),
    ("project_lighthouse_stack", [
        ("user", "Side project question -- I'm building a personal blog/portfolio site, calling it Lighthouse. Torn between Astro and Next.js for it. It's mostly static content, a handful of blog posts, no real interactivity."),
        ("assistant", "For mostly-static content with minimal interactivity, Astro is the stronger fit -- it ships zero JS by default and only hydrates the islands you explicitly mark, whereas Next.js carries more runtime weight you won't use for a static blog. If you ever need heavier client-side state later, Next.js scales up more easily, but that doesn't sound like Lighthouse's shape."),
        ("user", "Yeah that matches what I was leaning toward. Going with Astro for Lighthouse then -- the content collections feature looks like a good fit for the blog posts too."),
        ("assistant", "Good call. Astro's content collections give you type-safe frontmatter out of the box, which will save you some pain once you have more than a couple posts."),
    ]),
    ("pet_biscuit", [
        ("user", "Unrelated -- my dog Biscuit just turned six. He's a very anxious corgi who hates the mail carrier specifically."),
        ("assistant", "Happy birthday to Biscuit! Six years of mail-carrier vigilance is a real commitment. Corgis do tend to be alert little guys."),
    ]),
    ("hobby_climbing_intro", [
        ("user", "I've gotten really into bouldering the last year. I go with my friend Theo three times a week at a gym near my place."),
        ("assistant", "Three times a week is a serious habit, not just a hobby at that point. What's kept you and Theo coming back -- the physical challenge, the problem-solving side, the social part?"),
        ("user", "Honestly the problem-solving. Every route is basically a puzzle. Theo's way better than me though, he's been climbing for a decade."),
    ]),
    ("ephemeral_car_trouble", [
        ("user", "Ugh, my car wouldn't start this morning, had to get a jump from a neighbor and I'm going to be late to everything today."),
        ("assistant", "That's a rough start to the day. Was it the battery, or something else going on? If it's an older battery, might be worth getting it tested before it strands you somewhere less convenient."),
        ("user", "Probably the battery, it's been slow to turn over for a couple weeks. Going to deal with it this weekend, I hope."),
    ]),
    ("allergy_claim", [
        ("user", "Quick note for anything food-related -- I'm allergic to shellfish, so keep that in mind if you ever suggest restaurants or recipes."),
        ("assistant", "Noted, no shellfish. Thanks for flagging that up front -- I'll steer clear of it in any restaurant or recipe suggestions."),
    ]),
    ("book_recommendation", [
        ("user", "Just finished Project Hail Mary and honestly it might be my favorite book of the last few years."),
        ("assistant", "Project Hail Mary is a great pick -- Andy Weir really nails the problem-solving-under-pressure tone again. Was it the science puzzles or the Rocky relationship that got you most?"),
        ("user", "The Rocky relationship, no contest. Didn't expect to get that attached to it."),
    ]),
    ("general_knowledge_oauth", [
        ("user", "Can you explain how OAuth2's authorization code flow actually works end to end?"),
        ("assistant", "Sure -- the client redirects the user to the authorization server, the user logs in and consents, the server redirects back with a short-lived authorization code, and then the client exchanges that code (plus its client secret) for an access token via a back-channel request. The key security property is that the access token never passes through the browser, only the code does."),
        ("user", "Got it, that makes sense, thanks."),
    ]),
    ("ambiguous_alex_friend", [
        ("user", "My friend Alex from college is visiting next month. We haven't seen each other in person in like two years."),
        ("assistant", "That's exciting, two years is a long time between visits. Do you two have anything specific planned, or mostly just catching up?"),
        ("user", "Mostly catching up. Alex and I were roommates sophomore and junior year, so there's a lot of history there."),
    ]),
    ("time_bound_marathon", [
        ("user", "I'm training for a half marathon in Austin on October 18th. Currently about eight weeks out."),
        ("assistant", "Eight weeks out is a good spot to be building your long runs up steadily. How's the training going so far -- on pace, or feeling behind?"),
        ("user", "On pace I think. Longest run so far is nine miles, feeling okay."),
    ]),
    ("preference_coffee_diet", [
        ("user", "Random one -- I'm vegetarian, and my go-to coffee order is an oat milk cortado."),
        ("assistant", "Vegetarian, oat milk cortado -- noted. A cortado's a good choice if you want the espresso to actually come through instead of getting buried in milk."),
    ]),
    ("podcast_mention", [
        ("user", "I've been listening to a podcast called Search Engine on my runs lately, really well produced."),
        ("assistant", "Search Engine's a good one -- PJ Vogt's got a knack for picking questions that sound small and turning them into something surprisingly deep. What's been the best episode for you so far?"),
        ("user", "The one about why gas station coffee is bad, weirdly. Very not-small by the end."),
    ]),
    ("creative_writing_poem", [
        ("user", "Can you write me a short four-line poem about rain for a card I'm making?"),
        ("assistant", "Sure: \"The rain does not ask permission to fall, / it simply arrives and softens it all -- / the streets, the roofs, the tired afternoon light, / washed clean and quiet by morning's first sight.\""),
        ("user", "That's really nice, thank you."),
    ]),
    ("allergy_correction", [
        ("user", "Hey, actually I need to correct something I told you before -- I said I'm allergic to shellfish, but that's not quite right. I just really don't like shrimp, no actual allergy. My bad for the confusion."),
        ("assistant", "Good to know, thanks for the correction -- no allergy, just a shrimp dislike. I'll keep that distinction in mind, it does change what's actually off the table versus what's just a preference."),
    ]),
    ("hackathon_distractor", [
        ("user", "I'm thinking about signing up for a weekend web dev hackathon here in Austin. Never done one before."),
        ("assistant", "That could be a lot of fun, especially as a first one -- low stakes, high learning. Are you going in with a team already, or planning to find one there? Also worth thinking about ahead of time: do you want to optimize for finishing something polished, or for trying something you wouldn't normally get to touch at work?"),
        ("user", "No team yet, might just show up solo and see who I match with. Leaning toward trying something outside my usual backend comfort zone."),
        ("assistant", "Solo entry is a great way to meet people you'd never end up paired with otherwise. If you want to stretch outside backend, a lot of hackathon teams are short on whoever can wire up a working frontend fast, so that's often an easy way to pair up quickly."),
    ]),
    ("old_car_story", [
        ("user", "Saw a beat-up Honda Civic on the road today that looked exactly like my first car in college, a 2011 Civic I named Gerald for no good reason."),
        ("assistant", "Gerald the Civic is a great car name. Do you remember why 2011 Civics specifically stuck with you, or was Gerald just what you could afford at the time?"),
        ("user", "Definitely just what I could afford. But that car survived four years of very questionable driving decisions."),
    ]),
    ("childhood_sailing", [
        ("user", "Random memory that came up today -- I grew up sailing with my dad on weekends, small lake near where I grew up. Haven't done it in years though."),
        ("assistant", "That sounds like a nice thing to have had growing up. What made you stop -- just life getting busier, or moving away from the lake?"),
        ("user", "Mostly moving for college and then work. I keep meaning to get back into it someday."),
    ]),
    ("samuel_intro", [
        ("user", "My cousin Samuel is coming through Austin next week on a work trip, we're going to grab dinner."),
        ("assistant", "Nice, a work-trip dinner with family is always a good excuse to catch up. Is Samuel someone you see often, or is this more of a rare crossing-paths thing?"),
        ("user", "Pretty rare honestly, maybe once a year. He travels a ton for his job."),
    ]),
    ("move_new_city", [
        ("user", "Big update -- Jules and I are moving to Denver in a couple months. Mostly for Jules's job, but I'm excited about it too."),
        ("assistant", "That's a big change! Denver's a solid pick if you like being close to mountains. How are you feeling about leaving Austin after all this time here?"),
        ("user", "Mixed feelings. Austin's been home for a while, but change feels good right now. Going to miss the climbing gym here the most, honestly."),
    ]),
    ("ambiguous_alex_coworker", [
        ("user", "Started meeting some new people ahead of the Denver move -- there's an engineer at Nimbus named Alex Park who's going to be on my new team. Different Alex from my college friend, just a funny coincidence."),
        ("assistant", "Good coincidence to flag, that would get confusing fast otherwise. What's your first impression of Alex Park so far?"),
        ("user", "Seems sharp, asked good questions in the intro call. We'll see once I'm actually working with them day to day."),
    ]),
    ("job_change_nimbus", [
        ("user", "So the Denver move is actually because I took a new job -- I'm leaving Vantage Robotics and joining Nimbus Health as a senior backend engineer. Starts the same week we move."),
        ("assistant", "Congratulations, that's a meaningful jump -- new company, new domain, new city all at once. What pulled you toward Nimbus Health specifically?"),
        ("user", "The problem space, mostly. Health data infrastructure feels higher-stakes than warehouse robots in a good way. Bittersweet leaving Vantage after three years though."),
    ]),
    ("manager_callum", [
        ("user", "Met my new manager at Nimbus, Callum Reyes, in an onboarding call. Seems very direct, in a good way so far."),
        ("assistant", "Direct managers can be a nice change of pace if you like knowing exactly where you stand. How does Callum compare to Priya so far, or is it too early to tell?"),
        ("user", "Too early to tell really, but the directness reminds me a bit of Priya actually. Hoping that pattern holds."),
    ]),
    ("acknowledged_impostor_syndrome", [
        ("user", "I keep feeling like I got the Nimbus offer by luck, not because I'm actually qualified for a senior role, even though I know objectively that's not true."),
        ("assistant", "That gap between what you know objectively and what you feel has a name -- it's a pretty textbook case of impostor syndrome, and it's especially common right after a level-up like this one. It doesn't mean anything's actually wrong with your qualifications."),
        ("user", "Yeah, exactly, that's impostor syndrome. I've heard the term before but never really applied it to myself until just now."),
    ]),
    ("emotional_argument_roommate", [
        ("user", "Had a dumb argument with my old roommate Theo this week about who was supposed to handle the gym membership cancellation before I move. We sorted it out but it was annoying in the moment."),
        ("assistant", "Moving logistics have a way of surfacing exactly that kind of friction. Glad it got sorted out -- was it a quick resolution once you actually talked it through?"),
        ("user", "Yeah, five minute conversation once we both stopped being annoyed. Classic miscommunication, nothing serious."),
    ]),
    ("duolingo_streak", [
        ("user", "Hit a 400 day streak on Duolingo learning Spanish today, small win but I'm pretty proud of it."),
        ("assistant", "400 days is genuinely impressive, that's real consistency. How's your actual conversational Spanish coming along alongside the streak?"),
        ("user", "Slowly but surely. Reading's way ahead of speaking for me."),
    ]),
    ("fiddle_leaf_fig", [
        ("user", "My fiddle leaf fig finally pushed out a new leaf after basically sulking for two months. Small victories."),
        ("assistant", "Fiddle leaf figs are famously dramatic about that, so a new leaf is a real win. What ended up turning it around, more light, different watering, or just patience?"),
        ("user", "Honestly just patience and leaving it alone. I think I was fussing over it too much before."),
    ]),
    ("timbers_sports", [
        ("user", "Portland Timbers won again this weekend, my sister Nat's been sending me way too many celebratory texts."),
        ("assistant", "Nat sounds like a committed fan. Are you into the Timbers too, or is this more secondhand enthusiasm via your sister?"),
        ("user", "Mostly secondhand, but it's rubbing off on me a little if I'm honest."),
    ]),
    ("tamale_recipe", [
        ("user", "Made my grandmother's tamale recipe this weekend for the first time solo, no supervision. They actually turned out well."),
        ("assistant", "That's a real milestone, family recipes carry a lot of pressure to get right the first time alone. What was the trickiest part without supervision?"),
        ("user", "Getting the masa consistency right, definitely. Took two batches to get it where I remembered it should feel."),
    ]),
    ("samuel_callback", [
        ("user", "Sam texted me a photo from his latest work trip, this time in Singapore. That guy travels more than anyone I know."),
        ("assistant", "Singapore's a good one to add to the list. Is Sam's travel mostly client visits, or does his role just have him bouncing around constantly regardless of specific trips?"),
        ("user", "Pretty much constant regardless. I don't think he's been in one place for more than three weeks straight all year."),
    ]),
    ("hobby_climbing_callback", [
        ("user", "Going to miss climbing with Theo most out of everything once we're actually in Denver. We've found a new gym to check out there at least."),
        ("assistant", "Denver actually has a strong climbing scene, so that transition might go smoother than you'd expect even without Theo there. Have you looked at whether the new gym has a similar bouldering-heavy setup to what you're used to?"),
        ("user", "Yeah, looks pretty similar from photos. Still won't be the same without Theo roasting my technique though."),
    ]),
    ("unrelated_smalltalk", [
        ("user", "Hey, how's it going today?"),
        ("assistant", "Doing well, thanks for asking! How about you -- anything on your mind today, or just checking in?"),
        ("user", "Just checking in, nothing much going on. Kind of a slow day."),
    ]),
    ("dentist_appointment", [
        ("user", "Have a dentist appointment tomorrow morning that I'm dreading for no real reason, just general dentist-visit anxiety."),
        ("assistant", "That's a really common feeling even when there's nothing actually wrong. Is it a routine cleaning, or is there something specific prompting this one?"),
        ("user", "Just routine cleaning. I'll probably feel silly for dreading it once it's over, like always."),
    ]),
]


# Each probe is run against /read AFTER the full corpus above has been
# written. `expect` is a short human-readable note on what a correct
# retrieval should (or should NOT) contain -- read it before judging Qwen's
# output, don't just eyeball the raw block.
PROBES: list[tuple[str, str]] = [
    (
        "I'm thinking about signing up for that web dev hackathon next month, any tips on how to prep?",
        "REGRESSION CHECK: this echoes hackathon_distractor almost verbatim. The Astro/Lighthouse "
        "fact must NOT appear here -- if it does, that's the Karpathy-tweet failure mode (a "
        "tangential fact bleeding into an unrelated turn) and the graph is behaving like a flat "
        "summary instead of context-triggered recall.",
    ),
    (
        "What stack did I end up going with for Lighthouse again?",
        "Should surface Astro + the content-collections detail via the snippet, since Lighthouse "
        "is named explicitly here.",
    ),
    (
        "Where do I work these days, and who's my manager?",
        "Should surface Nimbus Health / Callum Reyes as CURRENT -- Vantage/Priya should be "
        "superseded, not returned as if still active (mutable WORKS_AT + one_to_one HAS_MANAGER "
        "supersede check).",
    ),
    (
        "Remind me about Alex.",
        "Ambiguity check: there are two distinct Alexes (college friend, Nimbus coworker Alex "
        "Park). Correct behavior is either surfacing both distinctly or the retriever asking-shaped "
        "context reflecting two separate nodes -- NOT one merged Alex with contradictory details.",
    ),
    (
        "Am I allergic to anything I should watch out for with food?",
        "Should reflect the CORRECTED state (shrimp dislike, no real allergy) -- not the original "
        "shellfish-allergy claim. Tests contradiction/supersede on a PREFERENCE-flavored fact.",
    ),
    (
        "How's my half marathon training going?",
        "Time-bound fact recall -- should mention the October 18th Austin half marathon and roughly "
        "the training status mentioned (nine mile long run).",
    ),
    (
        "What's the deal with my car?",
        "Ephemeral fact -- should surface the battery/jump-start incident. Note: real decay won't "
        "be visible yet in a same-day run; this is a retrieval-presence check, not a decay check.",
    ),
    (
        "Can you explain OAuth2 again?",
        "Should NOT surface any injected [MEMORY] block distinct from actual OAuth2 knowledge -- "
        "the earlier OAuth2 conversation was a pure Q&A with no durable content, so nothing "
        "graph-worthy should have been extracted from it in the first place.",
    ),
    (
        "What's my dietary preference for restaurant recommendations?",
        "Should surface vegetarian (a real durable preference), distinct from the "
        "shellfish/shrimp correction above -- two separate food-related facts, don't let one "
        "clobber the other.",
    ),
    (
        "Tell me about Sam.",
        "Abbreviation-merge check: 'Samuel' (cousin, intro'd by full name) and later 'Sam' (same "
        "person, casual) should resolve to ONE node, not two.",
    ),
    (
        "How's the climbing going, and who do I usually climb with?",
        "Should surface Theo + bouldering + the gym context, ideally reflecting both the original "
        "intro and the later Denver-move callback (co-episode / reinforcement check).",
    ),
    (
        "What was that book I couldn't stop talking about recently?",
        "Should surface Project Hail Mary, ideally with the Rocky-relationship detail from the "
        "snippet rather than just the title.",
    ),
    (
        "Do I have any pets?",
        "Should surface Biscuit the corgi.",
    ),
    (
        "What's going on with my living situation and family right now?",
        "Broad recall -- should pull together Jules (partner), the Denver move, and possibly Nat "
        "(sister) without needing any of those named explicitly in the query. This is the multi-hop "
        "/ spreading-activation stress test of the whole probe set.",
    ),
    (
        "Have I mentioned feeling unsure about myself at work lately?",
        "Should surface the impostor-syndrome conversation -- tests the 'acknowledged agent "
        "concept' extraction path (assistant named the concept, user affirmed it) rather than a "
        "verbatim keyword match.",
    ),
    (
        "What was that tricky bug I spent an afternoon chasing down at Vantage?",
        "Coding-session durability check: the surrounding stack trace / file path / error message "
        "noise should NOT have spawned garbage nodes, but the actual debugging lesson (desynced "
        "sensor clocks causing covariance drift) is exactly the kind of hard-won, one-off-but-worth-"
        "remembering fact the durability-threshold discussion is about. Under the CURRENT threshold "
        "it may well have been dropped as 'not durable' -- if so, that's a concrete before/after "
        "data point for the phase-2 threshold change, not a bug in this run.",
    ),
]
