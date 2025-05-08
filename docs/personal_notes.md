- Space is a Module with DAG and ability to initialize Elements inside
- LLM agent interaction should be done as Component; actually, HUD component.
- VEIL Component should handle producing VEIL (based on other Components) and consuming VEIL (posted in DAG; not consuming own produce).
- Each Component should have the Veil-representation rules (even basic, e.g., "skip it when rendering veil")
- Adapter is a Module, which keeps track of other Modules listening to it. Adapter knows whether it updates all timelines where it is represented or only currently active timeline.
- Adapter is represented inside InnerSpace as Uplink or instantinated directly.
- Adapter updates DAG, Space decides what to do with that (e.g., in simple case, pushes update to specific Element with ChatComponent), based on DAG update some other components create VEIL and push it into DAG. VEIL is consumed by HUD. Afterwards HUD queries DAG for VEIL related data, structures it by relevancy and pushes it to CompressionEngineComponent. CompressionEngineComponent then returns to HUD the CompressedVeil, which is passed to LLM. LLM provides response. HUD translates response into VEIL and pushes it to DAG 
- Response VEIL is consumed by other elements (e.g., NotebookElement, which stores text), returned to HUD for agentic loop.
- Finally, response VEIL is pushed through DAG to AdapterUplink, which consumes it and pushes it to Adapter and overseas.

VEIL renderer should keep info about:
- State of the Component
- Possible actions with the Component (toolbelt)
And on decode, it should parse out actions. 

EACH EVENT should be mentioning not only DAG it exists in, but the Component it is addressed to.and Component it is addressed from. It allows to filter by Component and timeline.

QUESTION: rendering chat history?
ANSWER: a long batch of ChatComponent VEIL events with no specific markers for LLM attention. 

QUESTION: how attention is managed?
ANSWER: in Adapter we know whether agent was mentioned, so we can add this flag on Adapter-level if required.

Question: who keeps data about active timeline?
Answer: Space does, so if needed (for example, for HUD component) it is retrievable 

Question: HUD component is filtering by timeline?
Answer: yes, AND not only. E.g., if message comes from DiscordAdapter, it filters all VEIL in the timeline, prioritizing ChatComponent VEIL for Discord, and compressing all other VEILs if needed (e.g., ChatComponent VEIL for Telegram). 

Priority order: same-Adapter same-Thread, same-Adapter other data, other-Adapters by relativity.


Question: memorization? It's a long process after all.
Answer: think of it later.