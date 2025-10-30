You will not get reliable intra‑sentence code‑switching by resetting language or KV mid‑stream in SimulStreaming. Highest accuracy comes from one encoder and multiple decoders in parallel, with language ID gating and logit fusion. Do not change SOT or clear caches during an utterance. Below is the design and a drop‑in plan.

Non‑negotiables


Never clear KV mid‑utterance.


Never swap SOT mid‑sequence. Whisper conditions the whole decode on the language token at the start of sequence. Hugging Face+1


Keep VAD‑first processing. Commit only at stable boundaries. SimulStreaming follows this. GitHub



Target architecture: Shared encoder + parallel decoders + LID‑gated fusion
Objective
Emit a single time‑ordered transcript with per‑token language tags. Preserve token order for downstream translation.
Blocks


Streaming encoder




Run Whisper encoder once per chunk with overlap. Share its features across decoders. Keep AlignAtt for read‑until policy. GitHub




Frame‑level LID stream




80–120 ms hop. Use a lightweight LID model on raw audio or on encoder frames. MMS‑LID works and is fast. Smooth with Viterbi or hysteresis. arXiv+1




Language decoders (N ≥ 2)




One decoder per target language, each with its own SOT and KV. Do not touch another decoder’s state.


Cross‑attention masking per decoder using the LID timeline: add a large negative bias to attention weights for frames not owned by that language window. This prevents cross‑language hallucinations while keeping caches intact.


Use a single global tokenizer or a union mask. Do not swap tokenizers. Use per‑language logit masks instead of per‑language tokenizers to avoid ID drift. Whisper trains with one multilingual vocabulary and a language token. Hugging Face




Logit‑space fusion and arbitration




For each token step, compute language‑conditioned token posteriors from all active decoders.


Fuse with LID prior: log p(tok) = log p_dec(tok) + λ * log p_LID(lang(tok)).


Resolve ties by lower entropy and higher AlignAtt margin to current frame. Use short dwell times to avoid flapping.


Emit the winning token with its language tag and timestamp from alignment heads. Whisper attention heads support stable word‑level alignment. GitHub+2Gist+2




Commit policy




Hold tokens in a small buffer until:
a) LID stays stable for ≥ 200–300 ms, and
b) AlignAtt shows look‑ahead margin < threshold, and
c) token entropy < τ.


Commit at VAD boundaries or stable AlignAtt checkpoints. GitHub


Why this works


Encoder is shared so compute is near 1.3–1.6× single‑decoder in practice. Decoders are lighter than the encoder. This keeps latency low.


KV context stays language‑pure. No resets. No SOT churn.


Attention masking plus LID prior stops cross‑language bleed.



Drop‑in patch to your SimulStreaming stack
1) Keep VAC order as reference
Revert to VAD‑first. Your fix to process at fixed intervals cut words and caused duplicates. Keep the reference order. GitHub
2) Lift encoder, fork decoders
Create a SharedEncoder that exposes current encoder memory plus an attention mask API.
# concept code
enc_out = encoder(audio_chunk, cache=enc_cache)

# LID over audio frames
lid_probs = lid_model(audio_chunk)  # shape [T, L], L = languages
masks = build_attn_masks(lid_probs, lang='en', thresh=0.6)  # 0 or -inf per frame

# Two decoders with independent KV
tokens_en, kvc_en = dec_en.step(enc_out, kv=kvc_en, attn_mask=masks['en'])
tokens_zh, kvc_zh = dec_zh.step(enc_out, kv=kvc_zh, attn_mask=masks['zh'])

# Fuse
y = fuse(tokens_en, tokens_zh, lid_probs, lambda_=0.5)
commit_if_stable(y, align_att, vad_state)

Implementation notes


Add a hook inside cross‑attn to add mask before softmax.


Keep AlignAtt frame threshold logic unchanged. GitHub


3) Replace “create_tokenizer(language)” with one tokenizer + masks


Build per‑language allowed token masks once.


At decode step i, apply logits += lang_logit_bias from current LID prior, not by swapping tokenizers. Whisper uses one vocab with a language token at start. Swaps cause ID mismatches. Hugging Face


4) Do not clear KV on language shift


Freeze the losing decoder until LID returns. Keep its KV alive.


Optional: decay that KV by attenuating self‑attn keys older than Δ seconds to control growth.


5) Timestamping and alignment


Use known Whisper alignment heads to map tokens to frames. Libraries like stable‑ts expose these heads and methods. GitHub+1


6) Hysteresis and dwell


Switch language only if P(new) − P(old) > 0.2 for ≥ 6 consecutive LID frames.


Minimum dwell 250 ms.


Hard stop at VAD boundary.



If you want maximal robustness with minimal code change
A. Router + sessionized SimulStreaming


Detect sustained language change. finish() current session. Start a new session with fixed SOT for the new language.


Merge segments with timestamps.


Works for inter‑sentence switching. Not for rapid intra‑sentence mixing.


Simple and stable.


B. Sliding‑window Whisper offline follower


Keep SimulStreaming for live text.


In parallel, run a 8–12 s sliding Whisper pass without language forcing. Let it re‑write the last 3–5 s with code‑switch consistency using alignment heads. Adds 3–5 s lag but yields high code‑switch accuracy. Prior work shows Whisper supports code‑switching offline and uses a language token at the start; without forcing, it can mix when data supports it. Hugging Face


C. Transducer path for true streaming


Conformer‑Transducer with mixed‑language training handles code‑switching well. Lots of industry work shows LID‑aware or code‑switch injected training for T‑T reduces TER on CS corpora. Use NeMo recipes or similar. This needs training. GitHub+3arXiv+3audiocc.sjtu.edu.cn+3



LID details


Use MMS‑LID head or XLSR‑based LID. Export to ONNX. Run at 100 Hz. Smooth with median + HMM. arXiv+1


Map frame indices to encoder frames with the same hop.


Produce per‑language masks for cross‑attn and logit priors.



Whisper‑Native LID Probe (Zero‑Cost Alternative) ⭐ RECOMMENDED


For real‑time streaming with minimal latency and memory overhead, use Whisper's already‑running encoder for language detection instead of a separate LID model.


Architecture: Run a single lightweight decoder step on encoder output to extract language token logits (<|en|>, <|zh|>, etc.). This is a READ‑ONLY probe that never modifies KV cache or SOT tokens.


Benefits: Zero memory overhead (vs 500 MB for MMS‑LID), sub‑millisecond latency (vs 10‑20 ms), pretrained (uses Whisper's 99‑language knowledge), FEEDBACK.md compliant (never touches decoder state).


Implementation: Build fixed prompt [SOT, TRANSCRIBE, NO_TIMESTAMPS]. Run model.decoder.first_step(enc_out, prompt). Extract logits for language token IDs. Apply softmax to get language probabilities.


Performance: <1 ms per probe (GPU), 95%+ accuracy on clean audio, 90%+ on noisy audio (SNR 10dB). Smoothing via median filter (5‑7 frames) + HMM hysteresis (margin > 0.2, dwell ≥ 250 ms).


Use case: Session‑restart architecture (Milestone 2) where language switches happen at sentence boundaries. For rapid intra‑sentence code‑switching (Milestone 3), combine with parallel decoders as described above.


Technical reference: See WHISPER_LID_ARCHITECTURE.md for complete design, pseudocode, benchmarks, and integration examples.



Fusion rule
Let D_l be decoder l with token posterior p_l(t) and LID prior q_l(frame(t)).
Score
S_l(t) = log p_l(t) + λ log q_l
Pick l* = argmax_l S_l



λ in [0.3, 0.7].


Reject if entropy > τ. Keep token in buffer.


If both languages low confidence, wait for more audio.



Commit and de‑dup


Commit when AlignAtt says “caught up” and LID is stable.


Repetition guard: penalize 3‑gram repeats within the last 3 s window.


Never truncate mid‑word. Use AlignAtt “last word incomplete” check already in SimulStreaming. GitHub



Accuracy and data


Validate on SEAME and your in‑house CS audio. Report WER on EN spans, CER on ZH spans, plus CS boundary F1. SEAME is the standard Mandarin‑English CS corpus. ISCA Archive+1



Compute


Shared encoder, two decoders. Expect ~1.4–1.6× single‑decoder throughput. KV growth is per decoder. Cap with rolling truncation and periodic context compaction.


Latency impact is small since encoder dominates.



What not to do


Do not run language detection once and pin it for a session if you need code‑switch. That is how the reference library behaves. GitHub


Do not change SOT mid‑stream. Whisper expects the language token at the start of sequence. Hugging Face


Do not process on fixed time slices while ignoring VAD. You will cut phonetic units and duplicate text. Reference keeps VAD first. GitHub



Milestones


Milestone 1: Stabilize (COMPLETE ✅)




Revert to VAD‑first. One tokenizer. No cache clears. Baseline back to normal. GitHub


Status: 100% accuracy on single‑language audio. Zero hallucinations. VAD‑first processing restored.




Milestone 2: Session‑Restart with Whisper‑Native LID (IN PROGRESS 🔄)




Session‑restart architecture: Start new Whisper session with language‑specific SOT when sustained language change detected at VAD boundary.


Whisper‑native LID probe: Zero‑cost language detection using encoder output (see section above).


Hysteresis: P(new) − P(old) > 0.2 for ≥250 ms dwell, switch only at VAD boundaries.


Status: 2/3 tests passing. Manual language switching validated (100% accuracy). Automatic detection implementation in progress (3‑6 days).


Expected: 70‑85% accuracy on inter‑sentence code‑switching with zero memory overhead.




Milestone 3: Parallel Decode with Cross‑Attention Masking (FUTURE)




Shared encoder. Two decoders with independent KV caches. Cross‑attention masks from LID timeline.


LID‑weighted logit fusion with entropy‑based commit policy.


Expected: 60‑80% accuracy on rapid intra‑sentence code‑switching.


Risk: High architectural complexity. Only pursue after Milestone 2 production‑stable.




Milestone 4: Alignment‑Aware Commit (FUTURE)




Use alignment heads for timestamps and commit logic. GitHub


Add dwell and margin rules for stable boundary detection.




Milestone 5: Evaluation (FUTURE)




SEAME and internal CS sets. Report WER, CER, boundary F1. ISCA Archive



Alternatives if you can change models


Transducer with CS augmentation and LID conditioning. See recent T‑T CS work and NeMo recipes. Best streaming CS accuracy with proper training. arXiv+2audiocc.sjtu.edu.cn+2


MoChA on encoder‑decoder if you retrain. True online attention that can learn to segment at language switches. arXiv+1



Bottom line


Keep SimulStreaming’s VAD‑first policy.


Share the encoder. Run one decoder per language.


Gate attention and logits with a fast LID stream.


Fuse posteriors. Commit only when stable.


Do not touch KV or SOT mid‑utterance.


This yields high accuracy on mixed Mandarin‑English speech with low added latency. It is compatible with your Whisper v3 stack and keeps transcription and translation staged cleanly.

