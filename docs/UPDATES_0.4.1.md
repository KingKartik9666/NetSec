# A few changes in NetSec 0.4.1

I’ve fixed the plain-English explanation failure and added a place to talk through the evidence.

- The local model no longer has to copy a paragraph word for word. It selects verified evidence points, and the app keeps the answer tied to those sources.
- There’s a new XAI chat page for follow-up questions about a result, influential features, uncertainty and what to check next.
- You can open a discussion from an analysis or a completed live window. Its evidence stays pinned while you talk.
- Answers show their sources. Chat stays focused on NetSec and cannot browse, run commands, change results or take actions on the network.
- Clear chat cancels an active reply. Everything remains local and is cleared when the app closes.

The existing theme, ten model choices and native CPU inference are retained. The current language model is still the small local Qwen model; the model-options guide lists possible upgrades separately. This Windows build remains unsigned.
