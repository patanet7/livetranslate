# Design-Clone Specification(Reference style: Supabase.com dark landing page)Target: “Visual Rules Builder” main landing page for a React/Next.js drag-and-drop rules-editor product.

---

##0. Global Design Language (Cloned from Reference)

1. Brand & Art Direction Overview • Ultra-minimal, developer-centric dark UI. • Matte-black (#0d0d0d) background, charcoal (#181818) cards, thin1px #2a2a2a borders, neon-green accent (#3FE37F) for CTAs & highlights. • Copy tone: concise, technical, confident; headlines split across two lines with second line accent-colored. • Subtle glass-HUD line-art illustrations & grids, faint star-sparkles.

2. Color Palette | Token | Hex | Usage | Notes | |-------------|-----------|-------------------------------------|----------------------------------| | bg-primary | #0d0d0d | Page background | same as reference | | card-bg | #181818 | Feature cards, nav dropdown | | | border-sub | #2a2a2a |1px borders around cards/inputs | | | text-main | #f1f1f1 | Primary copy | | | text-muted | #9e9e9e | Secondary copy | | | accent-green| #3FE37F | CTA buttons, headline highlight | identical hue to reference | | accent-white| #ffffff | Inverted text on green buttons | |3. Typography Scale • Headline XL:64/72px, weight600, line-height1.1 • Headline L:40/48px, weight600 • Body M:16/24px, weight400 • Caption:14/20px, weight400, upper-tracking0.05em • Font-family: “Inter”, system stack; same weights & kerning as reference.

4. Spacing & Layout Grid •12-col,72px max gutter,1440px max-width. • Vertical rhythm:96px section padding desktop,64px tablet,48px mobile. • Card grid:3-cols desktop (336px card),2-cols tablet,1-col mobile.

5. Visual Effects & Treatments • Cards:3px radius, border-sub stroke, inner-shadow0001px #000 inset. • Hover: border changes to accent-green + subtle elevation (0416rgba(0,0,0,.6)). • CTA buttons:4px radius, bold accent-green bg,120ms ease-in transform-scale1.03 on hover. • Subtle parallax sparkle SVG floating at5% opacity behind hero.

6. Component Styles • Top-nav:56px height, brand wordmark left, menu items center, “Dashboard” ghost button right. • Tab pills, badge labels, mega-menu, carousel arrows identical to reference.

---

##1. Project SummaryCreate the landing page for “RuleCanvas” – a visual drag-and-drop rules editor that lets developers sketch, connect, and deploy business logic instantly. Page must look indistinguishable from Supabase’s homepage in visual style, yet communicate RuleCanvas-specific value props.

---

##2. Main Page Overview (Route: `/`)Retain section order & structure:1. Navigation bar (dark)2. Hero with twin-line headline + dual CTAs + client logo strip3. Feature cards grid (6 cards)4. Success stories carousel5. “Start building in seconds” templates grid6. In-app screenshot section with tabs7. Slim footer (only if visible in reference; otherwise omit)All layout metrics, breakpoints, interactions mirror reference exactly; only copy & imagery change.

---

##3. Section-by-Section Specifications###3.1 Navigation Bar1. Visual Clone Instructions • Same56px matte-black bar,1px bottom border #2a2a2a. • Wordmark = lightning-bolt glyph + “RuleCanvas” text (white). • Menu items clone type, spacing: Product, Developers, Pricing, Docs, Blog. • Accent-green hover underline identical. • Right side: “Dashboard” pill button (#181818 bg, border #2a2a2a) + avatar circle.

2. Content Replacement • Brand name: RuleCanvas • Product submenu items: “Studio, Cloud Runtime, CLI, SDKs, Changelog, Pricing”. • Developers submenu identical structure but rename docs to “API Spec”.

3. Layout & Structure • Keep identical flex alignment,24px gap between nav items.4. Component Cloning • Full mega-menu with three columns, clone iconography layout.5. Asset Replacements • Wordmark glyph prompt: “Minimal lightning-bolt formed by two offset45° lines, neon-green on transparent background, flat icon.”6. Interaction Patterns • Same fade-in mega-menu, same highlight color on hover.

---

###3.2 Hero (“Build in a weekend” equivalent)

1. Visual Clone Instructions •640px tall center-stacked hero, headline split across two lines; second line accent-green. • Two pill buttons centered,8px gap. • Under buttons:6 client logos in1-row grayscale.

2. Content Replacement • Headline Line1: “Design rules in minutes” • Headline Line2 (accent-green): “Ship logic instantly” • Sub-text (same length): “RuleCanvas lets you craft, test, and deploy dynamic business rules with a visual node editor built for modern stacks.” • Primary CTA: “Open the editor” (accent-green filled) • Secondary CTA: “Watch a3-min demo” (ghost) • Client logos: Stripe, Shopify, Notion, Vercel, GitHub, Netlify (white-on-grey SVGs).

3. Layout & Structure • Same max-widths,32px top margin headline to sub-text.4. Component Cloning • Buttons identical styling/interaction.5. Asset Replacements • Each logo prompt: “Flat monochrome white logo of [company] on transparent background.”

6. Interaction Patterns • Hover scale + slight inner-shadow identical.

---

###3.3 Feature Cards Grid (6-up)

1. Visual Clone Instructions • Clone3×2 card matrix, equal card sizing,24px gaps,3px radius, border-sub. • Icon or micro-illustration top-left of each card (grey line-art). • Checkbox list or code snippet area identical placement.

2. Content Replacement (Keep caption & bullet counts identical) • Card1 Title: “Visual Node Editor” – Bullets: ✓ Drag-and-drop nodes ✓ Zoom, pan, snap-to-grid ✓ Keyboard shortcuts • Card2: “Real-time Validation” – show3 input rows with colored status pills. • Card3: “Edge Deploy” – include vector grid globe illustration replaced by “rule-deployment arc diagram”. • Card4: “Version Control” – show commit icons. • Card5: “Team Collaboration” – chat bubbles diagram. • Card6: “Language-agnostic Runtime” – mock code window listing “npm i rulecanvas”.

3. Layout & Structure • Same internal paddings (32px).4. Component Cloning • Copy check-list style, code-block shell, corner badge if any.

5. Asset Replacements • Provide prompts per card illustration, e.g., “Isometric line-art of interconnected squares forming a flowchart, monochrome #4a4a4a on #181818.”6. Interaction Patterns • Hover border accent-green & slight lift identical.

---

###3.4 Success Stories Carousel (“Infrastructure to innovate”)

1. Visual Clone Instructions • Horizontal scroll list of customer cards, identical size & border. • Section header small overline + H2, flush left.

2. Content Replacement • Overline: “CUSTOMER STORIES” • Header: “Simplifying logic at scale.” • Sub-copy: “See how engineering teams accelerate feature delivery using RuleCanvas.” • Buttons: “All stories” (accent-green) + “Submit story” (ghost). • Cards (6): Quantify results: – “Shipyard” – “Reduced rule bugs by83% in one sprint.” – “Kwarky” – etc. Maintain same char length.

3. Asset Replacements • Each card icon prompt: “Monochrome white logotype of [fictional brand] on transparent background.”

---

###3.5 Templates Grid (“Start building in seconds”)

1. Visual Clone Instructions •2-row card grid identical to reference (same spacing).2. Content Replacement • H2: “Kickstart with ready-made templates” • Sub-copy: “Clone community blueprints to speed up your workflow.” • Two top buttons unchanged style (“All templates”, “GitHub repo”). • Cards (6) titles:1. “Role-based Access Rules”2. “E-commerce Discounts Flow”3. “Slack Approval Bot”4. “IoT Device Throttling”5. “A/B Testing Logic”6. “Subscription Renewal Hooks” • Each description same length as reference.3. Asset Replacements • Each card icon group prompt example: “Minimalist circle icon with letter ‘RB’ in white, on #181818, style matching Supabase template icons.”

---

###3.6 In-App Screenshot Section (“Stay productive and manage your app”)

1. Visual Clone Instructions • Centered mock browser window, tabs row above screenshot, grey frame, identical padding. • Overline/pills row above with4 option pills.2. Content Replacement • Section header: Line1 (white): “Build, test, deploy” Line2 (muted): “without leaving the canvas” • Pills: “Rule Editor (active), Run Test, Versions, Docs”. • Tick-list under pills: replicate6 checkmarks but text: Full Undo/Redo, Live Preview, … (same length). • Screenshot: replace table editor with RuleCanvas interface mock showing nodes “Input → Transform → Output” in dark theme.

3. Asset Replacement Prompt • “720×420 PNG of a dark-theme flow-based editor, nodes connected by glowing green lines on a grid background, minimalist icons, looks like Supabase UI.”---

###3.7 Footer (if cloning)

1. Visual Clone Instructions • Same dark footer,3-column links, right-aligned social icons.

2. Content Replacement • Replace link groups with: Product, Resources, Community. • Copyright: “©2024 RuleCanvas Inc.”

---

## Clone Fidelity Checklist✓ All colors, fonts, sizes identical to reference.✓ New copy matches line/character counts to preserve wrap & rhythm.✓ Section order, grid counts, hover/transition effects cloned1-for-1.✓ Every illustration & logo replaced via detailed prompts while preserving style.✓ No original Supabase wording or brand appears; all content aligned with drag-and-drop rules editor use-case.